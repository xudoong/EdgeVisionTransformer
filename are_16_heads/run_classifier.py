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
"""DeiT finetuning runner."""

import os
import random
import tempfile

import numpy as np
import torch
from torch.optim import SGD
from torch.nn.parallel import DistributedDataParallel as DDP

import classifier_args
# import classifier_data as data
from logger import logger
import pruning
from classifier_eval import (
    evaluate,
    calculate_head_importance,
)
import classifier_training as training
from classifier_scoring import Accuracy
from util import get_vit_config, build_dataset


def prune_heads_plus_ddp(model, to_prune):
    if hasattr(model, 'module'):
        model.module.vit.prune_heads(to_prune)
        model = DDP(model.module)
    else:
        model.vit.prune_heads(to_prune)
    return model

def main():
    # Arguments
    parser = classifier_args.get_base_parser()
    classifier_args.training_args(parser)
    classifier_args.fp16_args(parser)
    classifier_args.pruning_args(parser)
    classifier_args.eval_args(parser)
    classifier_args.analysis_args(parser)
    classifier_args.export_onnx_args(parser)
    classifier_args.finetune_args(parser)

    args = parser.parse_args()

    # ==== CHECK ARGS AND SET DEFAULTS ====

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

    if (args.n_retrain_steps_after_pruning  or args.n_retrain_epochs_after_pruning) and args.retrain_pruned_heads:
        raise ValueError(
            "(--n_retrain_steps_after_pruning or --n_retrain_epochs_after_pruning) and --retrain_pruned_heads are "
            "mutually exclusive"
        )
    if torch.cuda.device_count() > 1 and args.export_onnx:
        raise ValueError(
            'Only allowed to export_onnx without data parallelism.'
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
        if not args.use_huggingface_trainer:
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



    # ==== PREPARE DATA ====
    # Train data
    if args.do_train or args.do_prune:
        # Prepare training data
        if args.dry_run:
            pass # TODO
        else:
            train_dataset, _ = build_dataset(args.data_dir, is_train=True, shuffle=True, return_dict=args.use_huggingface_trainer)


    # Eval data
    if args.do_eval or (args.do_prune and args.eval_pruned):
        if args.dry_run:
            pass # TODO
        else:
            eval_dataset, _ = build_dataset(args.data_dir, is_train=False, shuffle=False, return_dict=False)


    # ==== PREPARE MODEL ====
    def get_deit_model():
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained(f'facebook/deit-{args.deit_type}-patch16-224')

        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1 and not args.use_huggingface_trainer:
            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
        
        return model

    model = get_deit_model()


    # ==== PREPARE TRAINING ====

    if args.use_huggingface_trainer:
        training_args = training.get_training_args(
            learning_rate=args.learning_rate,
            micro_batch_size=args.train_batch_size,
            n_steps=args.n_retrain_steps_after_pruning,
            n_epochs=args.n_retrain_epochs_after_pruning,
            local_rank=args.local_rank,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
        )
        # initialize TrainingArguments twice would result in error: dist.init_process_group twice!
        # finetune_args = training.get_training_args(
        #     learning_rate=args.finetune_learning_rate,
        #     micro_batch_size=args.train_batch_size,
        #     n_steps=None,
        #     n_epochs=args.n_retrain_epochs_after_pruning,
        #     local_rank=args.local_rank,
        #     num_workers=args.num_workers,
        #     output_dir=args.output_dir,
        # )
        import copy
        finetune_args = copy.deepcopy(training_args)
        finetune_args.n_steps = args.n_finetune_steps_after_pruning or -1
        finetune_args.n_epochs = args.n_finetune_epochs_after_pruning or -1

    is_main = args.local_rank == -1 or args.local_rank == 0
    
    # ==== PRUNE ====
    retrain_model_save_path_tem = os.path.join(args.output_dir, f'deit_{args.deit_type}_are16heads_prune{{num_pruned_heads}}_retrained.state_dict')
    finetune_model_save_path_tem = os.path.join(args.output_dir, f'deit_{args.deit_type}_are16heads_prune{{num_pruned_heads}}{{retrain_flag}}_finetuned.state_dict')
    if args.do_prune:
        if args.fp16:
            raise NotImplementedError("FP16 is not yet supported for pruning")

        # Determine the number of heads to prune
        prune_sequence = pruning.determine_pruning_sequence(
            args.prune_number,
            args.prune_percent,
            get_vit_config(model).num_hidden_layers,
            get_vit_config(model).num_attention_heads,
            args.at_least_x_heads_per_layer,
        )
        # Prepare optimizer for tuning after pruning
        if args.n_retrain_steps_after_pruning or args.n_retrain_epochs_after_pruning:
            retrain_optimizer = SGD(
                model.parameters(),
                lr=args.retrain_learning_rate
            )
        # Prepare optimizer for finetuning
        if args.n_finetune_epochs_after_pruning or args.n_finetune_steps_after_pruning:
            finetune_optimizer = SGD(
                model.parameters(),
                lr=args.finetune_learning_rate
            )

        to_prune = {}
        for step, n_to_prune in enumerate(prune_sequence):
            if is_main:
                logger.info('====================================================================================================================')
                logger.info('====================================================================================================================')
            
            if step == 0 or args.exact_pruning:
                # Calculate importance scores for each layer
                if step == 0 and args.head_importance_file:
                    # load txt file
                    assert(args.head_importance_file.endswith('.txt'))
                    if is_main:
                        logger.info(f'Load head_importance_score from {args.head_importance_file}')
                    head_importance = torch.from_numpy(np.loadtxt(args.head_importance_file, dtype=np.float32))
                else:
                    head_importance = calculate_head_importance(
                        model,
                        train_dataset,
                        batch_size=args.train_batch_size,
                        device=device,
                        normalize_scores_by_layer=args.normalize_pruning_by_layer,
                        subset_size=args.compute_head_importance_on_subset,
                        verbose=True,
                        disable_progress_bar=args.no_progress_bars,
                        distributed=args.local_rank != -1,
                        num_workers=args.num_workers,
                        pruned_heads=to_prune
                    )
                if is_main:
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
            num_pruned_heads =  sum(len(heads) for heads in to_prune.values())
            # Actually mask the heads
            if args.actually_prune:
                model = prune_heads_plus_ddp(model, to_prune)
            else:
                model.vit.mask_heads(to_prune)

            if is_main:
                logger.info('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
            # Maybe continue training a bit
            retrain_model_save_path = retrain_model_save_path_tem.format(num_pruned_heads=num_pruned_heads)
            if args.n_retrain_steps_after_pruning or args.n_retrain_epochs_after_pruning:
                set_seeds(args.seed + step + 1, n_gpu)
                if args.use_huggingface_trainer:
                    training.huggingface_trainer_train(
                        train_dataset=train_dataset,
                        model=model,
                        args=training_args
                    )
                else:
                    training.train(
                        train_dataset,
                        model,
                        retrain_optimizer,
                        args.train_batch_size,
                        n_steps=args.n_retrain_steps_after_pruning,
                        n_epochs=args.n_retrain_epochs_after_pruning,
                        device=device,
                        verbose=True,
                        local_rank=args.local_rank,
                        num_workers=args.num_workers,
                    )
                # save model
                if is_main:
                    state_dict = {
                        'model': getattr(model, 'module', model).state_dict(),
                        'num_pruned_heads':num_pruned_heads,
                    }
                    torch.save(state_dict, retrain_model_save_path)
                    logger.info(f'Save retrained model to {retrain_model_save_path}')
            
            if args.export_onnx:
                from utils import export_onnx
                total_pruned = sum(len(heads) for heads in to_prune.values())
                export_onnx(torch_model=model.to(torch.device('cpu')),
                            output_path=os.path.join(args.onnx_output_dir, f'deit_{args.deit_type}_are16heads_prune{total_pruned}.onnx'),
                            input_shape=[1,3,224,224],
                            dynamic_batch=True)

            if is_main:
                logger.info('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
            # Evaluate
            if args.eval_pruned:
                # Print the pruning descriptor
                if is_main:
                    logger.info("Evaluating following pruning strategy")
                    logger.info(pruning.to_pruning_descriptor(to_prune))
                # Eval accuracy
                scorer = Accuracy()
                accuracy = evaluate(
                    eval_dataset,
                    model,
                    args.eval_batch_size,
                    save_attention_probs=args.save_attention_probs,
                    print_head_entropy=False,
                    device=device,
                    verbose=False,
                    disable_progress_bar=args.no_progress_bars,
                    scorer=scorer,
                    distributed=args.local_rank != -1,
                    num_workers=args.num_workers
                )[scorer.name]

                if is_main:
                    logger.info("***** Pruning eval results *****")
                    tot_pruned = sum(len(heads) for heads in to_prune.values())
                    logger.info(f"{tot_pruned}\t{accuracy}")

                    if (args.n_retrain_steps_after_pruning or args.n_retrain_epochs_after_pruning) and is_main:
                        state_dict = torch.load(retrain_model_save_path)
                        state_dict['accuracy'] = accuracy
                        torch.save(state_dict, retrain_model_save_path)


            if is_main:
                logger.info('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
            # finetune
            finetune_model_save_path = finetune_model_save_path_tem.format(
                num_pruned_heads=num_pruned_heads, 
                retrain_flag='_retrained' if args.n_retrain_steps_after_pruning or args.n_retrain_epochs_after_pruning else ''
            )
            if args.n_finetune_epochs_after_pruning or args.n_finetune_steps_after_pruning:
                if is_main:
                    logger.info("***** Running Fine-tuning *****")
                state_dict_before_finetune = {
                    'model': getattr(model, 'module', model).state_dict()
                }
                set_seeds(args.seed + step + 1, n_gpu)
                if args.use_huggingface_trainer:
                    training.huggingface_trainer_train(
                        train_dataset=train_dataset,
                        model=model,
                        args=finetune_args
                    )
                else:
                    training.train(
                        train_dataset,
                        model,
                        finetune_optimizer,
                        args.train_batch_size,
                        n_steps=args.n_finetune_steps_after_pruning,
                        n_epochs=args.n_finetune_epochs_after_pruning,
                        device=device,
                        verbose=True,
                        local_rank=args.local_rank,
                        num_workers=args.num_workers
                    )

                state_dict_after_finetune = {
                    'model': getattr(model, 'module', model).state_dict()
                }

                if is_main:
                    logger.info('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')

                if args.eval_finetuned:
                    # Print the pruning descriptor
                    if is_main:
                        logger.info("Evaluating following pruning strategy after finetuning")
                        logger.info(pruning.to_pruning_descriptor(to_prune))
                    # Eval accuracy
                    scorer = Accuracy()
                    accuracy = evaluate(
                        eval_dataset,
                        model,
                        args.eval_batch_size,
                        save_attention_probs=args.save_attention_probs,
                        print_head_entropy=False,
                        device=device,
                        verbose=False,
                        disable_progress_bar=args.no_progress_bars,
                        scorer=scorer,
                        distributed=args.local_rank != -1,
                        num_workers=args.num_workers
                    )[scorer.name]

                    if is_main:
                        logger.info("***** Finetuning eval results *****")
                        tot_pruned = sum(len(heads) for heads in to_prune.values())
                        logger.info(f"Finetuned {tot_pruned}\t{accuracy}")

                        state_dict_after_finetune['accuracy'] = accuracy
                        torch.save(state_dict_after_finetune, finetune_model_save_path)
                        logger.info(f'Save finetuned model to {finetune_model_save_path}')

                model.load_state_dict(state_dict_before_finetune['model'])
                if is_main:
                    logger.info('Loaded back the model before finetuning.')


if __name__ == "__main__":
    main()
