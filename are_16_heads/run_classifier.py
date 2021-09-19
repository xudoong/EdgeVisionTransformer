# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc.
# team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import os
import random
import tempfile

import numpy as np
import torch
from torch.optim import SGD

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import (
    BertForSequenceClassification,
    BertConfig,
)
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import classifier_args
import classifier_data as data
from logger import logger
import pruning
from classifier_eval import (
    evaluate,
    calculate_head_importance,
    analyze_nli,
    predict,
)
import classifier_training as training


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def prepare_dry_run(args):
    args.no_cuda = True
    args.train_batch_size = 3
    args.eval_batch_size = 3
    args.do_train = True
    args.do_eval = True
    args.do_prune = True
    args.do_anal = False
    args.output_dir = tempfile.mkdtemp()
    return args


def main():
    # Arguments
    parser = classifier_args.get_base_parser()
    classifier_args.training_args(parser)
    classifier_args.fp16_args(parser)
    classifier_args.pruning_args(parser)
    classifier_args.eval_args(parser)
    classifier_args.analysis_args(parser)

    args = parser.parse_args()

    # ==== CHECK ARGS AND SET DEFAULTS ====

    if args.dry_run:
        args = prepare_dry_run(args)

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: "
            f"{args.gradient_accumulation_steps}, should be >= 1"
        )

    args.train_batch_size = int(
        args.train_batch_size
        / args.gradient_accumulation_steps
    )

    if not (args.do_train or args.do_eval or args.do_prune or args.do_anal):
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_prune` must be True."
        )
    out_dir_exists = os.path.exists(args.output_dir) and \
        os.listdir(args.output_dir)
    if out_dir_exists and args.do_train and not args.overwrite:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not "
            "empty."
        )

    if args.n_retrain_steps_after_pruning > 0 and args.retrain_pruned_heads:
        raise ValueError(
            "--n_retrain_steps_after_pruning and --retrain_pruned_heads are "
            "mutually exclusive"
        )

    # ==== SETUP DEVICE ====

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda
            else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        f"device: {device} n_gpu: {n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.fp16}"
    )

    # ==== SETUP EXPERIMENT ====

    def set_seeds(seed, n_gpu):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    set_seeds(args.seed, n_gpu)

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in data.processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = data.processors[task_name]()
    num_labels = data.num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    # ==== PREPARE DATA ====

    # Train data
    if args.do_train or args.do_prune:
        # Prepare training data
        if args.dry_run:
            train_examples = processor.get_dummy_train_examples(args.data_dir)
        else:
            train_examples = processor.get_train_examples(args.data_dir)
        train_data = data.prepare_tensor_dataset(
            train_examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            verbose=args.verbose,
        )

    # Eval data
    if args.do_eval or (args.do_prune and args.eval_pruned):
        if args.dry_run:
            eval_examples = processor.get_dummy_dev_examples(args.data_dir)
        else:
            eval_examples = processor.get_dev_examples(args.data_dir)
        # data.add_dependency_arcs(eval_examples)
        # print(eval_examples[-2].parse_a)
        eval_data = data.prepare_tensor_dataset(
            eval_examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            verbose=args.verbose,
        )

    # ==== PREPARE MODEL ====

    def get_model(
        model_type,
        toy_classifier=False,
        dry_run=False,
        n_heads=1,
        state_dict=None,
        cache_dir=None
    ):

        if dry_run:
            model = BertForSequenceClassification(
                BertConfig.dummy_config(len(tokenizer.vocab)),
                num_labels=num_labels
            )
        else:
            model = BertForSequenceClassification.from_pretrained(
                model_type,
                cache_dir=cache_dir,
                num_labels=num_labels,
                state_dict=None if toy_classifier else state_dict,
            )
        if toy_classifier:
            config = BertConfig(
                len(tokenizer.vocab),
                hidden_size=768,
                num_hidden_layers=1,
                num_attention_heads=n_heads,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02
            )
            toy_model = BertForSequenceClassification(
                config,
                num_labels=num_labels
            )
            toy_model.bert.embeddings.load_state_dict(model.bert.embeddings.state_dict())
            if state_dict is not None:
                model_to_load = getattr(toy_model, "module", toy_model)
                model_to_load.load_state_dict(state_dict)
            model = toy_model

        return model

    model = get_model(
        args.bert_model,
        toy_classifier=args.toy_classifier,
        dry_run=args.dry_run,
        n_heads=args.toy_classifier_n_heads,
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
        f"distributed_{args.local_rank}",
    )
    # Head dropout
    for layer in model.bert.encoder.layer:
        layer.attention.self.dropout.p = args.attn_dropout

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex "
                "to use distributed and fp16 training."
            )

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Parse pruning descriptor
    to_prune = pruning.parse_head_pruning_descriptors(
        args.attention_mask_heads,
        reverse_descriptors=args.reverse_head_mask,
    )
    # Mask heads
    if args.actually_prune:
        model.bert.prune_heads(to_prune)
    else:
        model.bert.mask_heads(to_prune)


    # ==== PREPARE TRAINING ====

    # Trainable parameters
    if args.do_train or args.do_prune:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # Only train the classifier in feature mode
        if args.feature_mode:
            param_optimizer = [(n, p) for n, p in param_optimizer
                               if n.startswith("classifier")]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # Prepare optimizer for fine-tuning on task
    if args.do_train:
        num_train_steps = int(
            len(train_examples)
            / args.train_batch_size
            / args.gradient_accumulation_steps
        ) * args.num_train_epochs
        optimizer, lr_schedule = training.prepare_bert_adam(
            optimizer_grouped_parameters,
            args.learning_rate,
            num_train_steps,
            args.warmup_proportion,
            loss_scale=args.loss_scale,
            local_rank=args.local_rank,
            fp16=args.fp16,
        )

    # ==== TRAIN ====
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        # Train
        global_step, tr_loss, nb_tr_steps = training.train(
            train_data,
            model,
            optimizer,
            args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=device,
            verbose=True,
            disable_progress_bar=args.no_progress_bars,
            n_gpu=n_gpu,
            global_step=global_step,
            lr_schedule=lr_schedule,
            n_epochs=args.num_train_epochs,
            local_rank=args.local_rank,
            fp16=args.fp16,
        )

    # Save train loss
    result = {"global_step": global_step,
              "loss": tr_loss/nb_tr_steps if args.do_train else None}

    # Save a trained model
    # Only save the model it-self
    model_to_save = getattr(model, "module", model)
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model = get_model(
        args.bert_model,
        toy_classifier=args.toy_classifier,
        dry_run=args.dry_run,
        n_heads=args.toy_classifier_n_heads,
        state_dict=model_state_dict,
    )
    model.to(device)

    is_main = args.local_rank == -1 or torch.distributed.get_rank() == 0

    # Parse pruning descriptor
    to_prune = pruning.parse_head_pruning_descriptors(
        args.attention_mask_heads,
        reverse_descriptors=args.reverse_head_mask,
    )
    # Mask heads
    if args.actually_prune:
        model.bert.prune_heads(to_prune)
    else:
        model.bert.mask_heads(to_prune)

    # ==== PRUNE ====
    if args.do_prune and is_main:
        if args.fp16:
            raise NotImplementedError("FP16 is not yet supported for pruning")

        # Determine the number of heads to prune
        prune_sequence = pruning.determine_pruning_sequence(
            args.prune_number,
            args.prune_percent,
            model.bert.config.num_hidden_layers,
            model.bert.config.num_attention_heads,
            args.at_least_x_heads_per_layer,
        )
        # Prepare optimizer for tuning after pruning
        if args.n_retrain_steps_after_pruning > 0:
            retrain_optimizer = SGD(
                model.parameters(),
                lr=args.retrain_learning_rate
            )
        elif args.retrain_pruned_heads:
            if args.n_retrain_steps_pruned_heads > 0:
                num_retrain_steps = args.n_retrain_steps_pruned_heads
            else:
                num_retrain_steps = int(
                    len(train_examples)
                    / args.train_batch_size
                    / args.gradient_accumulation_steps
                ) * args.num_train_epochs

        to_prune = {}
        for step, n_to_prune in enumerate(prune_sequence):

            if step == 0 or args.exact_pruning:
                # Calculate importance scores for each layer
                head_importance = calculate_head_importance(
                    model,
                    train_data,
                    batch_size=args.train_batch_size,
                    device=device,
                    normalize_scores_by_layer=args.normalize_pruning_by_layer,
                    subset_size=args.compute_head_importance_on_subset,
                    verbose=True,
                    disable_progress_bar=args.no_progress_bars,
                )
                logger.info("Head importance scores")
                for layer in range(len(head_importance)):
                    layer_scores = head_importance[layer].cpu().data
                    logger.info("\t".join(f"{x:.5f}" for x in layer_scores))
            # Determine which heads to prune
            to_prune = pruning.what_to_prune(
                head_importance,
                n_to_prune,
                to_prune={} if args.retrain_pruned_heads else to_prune,
                at_least_x_heads_per_layer=args.at_least_x_heads_per_layer
            )
            # Actually mask the heads
            if args.actually_prune:
                model.bert.prune_heads(to_prune)
            else:
                model.bert.mask_heads(to_prune)
            # Maybe continue training a bit
            if args.n_retrain_steps_after_pruning > 0:
                set_seeds(args.seed + step + 1, n_gpu)
                training.train(
                    train_data,
                    model,
                    retrain_optimizer,
                    args.train_batch_size,
                    n_steps=args.n_retrain_steps_after_pruning,
                    device=device,
                )
            elif args.retrain_pruned_heads:
                set_seeds(args.seed + step + 1, n_gpu)
                # Reload BERT
                base_bert = None
                if args.reinit_from_pretrained:
                    base_bert = BertForSequenceClassification.from_pretrained(  # noqa
                        args.bert_model,
                        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
                        f"distributed_{args.local_rank}",
                        num_labels=num_labels
                    ).bert
                    base_bert.to(device)
                # Reinit
                model.bert.reset_heads(to_prune, base_bert)
                # Unmask heads
                model.bert.clear_heads_mask()
                if args.only_retrain_val_out:
                    self_att_params = [
                        p for layer in model.bert.encoder.layer
                        for p in layer.attention.self.value.parameters()
                    ]
                else:
                    self_att_params = [
                        p for layer in model.bert.encoder.layer
                        for p in layer.attention.self.parameters()
                    ]
                head_grouped_parameters = {
                    'params':
                        self_att_params +
                        [p for layer in model.bert.encoder.layer
                         for p in layer.attention.output.dense.parameters()],
                    'weight_decay': 0.01
                }
                retrain_optimizer, lr_schedule = training.prepare_bert_adam(
                    [head_grouped_parameters],
                    args.learning_rate,
                    num_retrain_steps,
                    args.warmup_proportion,
                    loss_scale=args.loss_scale,
                    local_rank=args.local_rank,
                    fp16=args.fp16,
                )
                training.train(
                    train_data,
                    model,
                    retrain_optimizer,
                    args.train_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,  # noqa
                    device=device,
                    verbose=True,
                    disable_progress_bar=args.no_progress_bars,
                    n_gpu=n_gpu,
                    global_step=0,
                    lr_schedule=lr_schedule,
                    n_epochs=args.num_train_epochs,
                    local_rank=args.local_rank,
                    fp16=args.fp16,
                    mask_heads_grad=to_prune,
                    n_steps=num_retrain_steps,
                    eval_mode=args.no_dropout_in_retraining,
                )

            # Evaluate
            if args.eval_pruned:
                # Print the pruning descriptor
                logger.info("Evaluating following pruning strategy")
                logger.info(pruning.to_pruning_descriptor(to_prune))
                # Eval accuracy
                metric = processor.scorer.name
                accuracy = evaluate(
                    eval_data,
                    model,
                    args.eval_batch_size,
                    save_attention_probs=args.save_attention_probs,
                    print_head_entropy=True,
                    device=device,
                    verbose=True,
                    disable_progress_bar=args.no_progress_bars,
                    scorer=processor.scorer,
                )[metric]
                logger.info("***** Pruning eval results *****")
                tot_pruned = sum(len(heads) for heads in to_prune.values())
                logger.info(f"{tot_pruned}\t{accuracy}")

    # ==== EVALUATE ====
    if args.do_eval and is_main:
        evaluate(
            eval_data,
            model,
            args.eval_batch_size,
            save_attention_probs=args.save_attention_probs,
            print_head_entropy=False,
            device=device,
            result=result,
            disable_progress_bar=args.no_progress_bars,
            scorer=processor.scorer,
        )
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    # ==== ANALYZIS ====
    if args.do_anal:
        if not data.is_nli_task(processor):
            logger.warn(
                f"You are running analysis on the NLI diagnostic set but the "
                f"task ({args.task_name}) is not NLI"
            )
        anal_processor = data.DiagnosticProcessor()
        if args.dry_run:
            anal_examples = anal_processor.get_dummy_dev_examples(
                args.anal_data_dir)
        else:
            anal_examples = anal_processor.get_dev_examples(args.anal_data_dir)
        anal_data = data.prepare_tensor_dataset(
            anal_examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            verbose=args.verbose,
        )
        predictions = predict(
            anal_data,
            model,
            args.eval_batch_size,
            verbose=True,
            disable_progress_bar=args.no_progress_bars,
            device=device,
        )
        report = analyze_nli(anal_examples, predictions, label_list)
        # Print report
        for feature, values in report.items():
            print("=" * 80)
            print(f"Scores breakdown for feature: {feature}")
            for value, accuracy in values.items():
                print(f"{value}\t{accuracy:.5f}")


if __name__ == "__main__":
    main()
