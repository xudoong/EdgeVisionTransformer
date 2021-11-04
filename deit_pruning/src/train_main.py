from sys import maxsize
from warnings import showwarning
from torch import optim
from torch.nn.functional import alpha_dropout
from torch.optim import optimizer
from tqdm.utils import disp_trim
from transformers import Trainer, TrainingArguments, AutoModelForImageClassification
import transformers
from model import SwiftBERT
from data import AdDataset, AdIterableDataset, ShuffleDataset, rawgencount
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
import json
import os

from utils import show_deit_sparsity, swift_converter, set_random, unzero_parameters, build_dataset, evaluate, dist_print

from trainer import SparseWithoutTeacherTrainer, SparserWithTeacherTrainer, TrainerWithTeacher, DistilTrainingArguments
from nn_pruning.patch_coordinator import ModelPatchingCoordinator, SparseTrainingArguments
from nn_pruning.inference_model_patcher import optimize_model


def sparse_argument_builder(args):
    def set_dict(dict, item, name):
        if item is not None:
            dict[name] = item
    #print('build sparse',args.attention_threshold)
    print(args)
    if not args.sparse_preset:

        #print('here',attention_threshold, args.attention_threshold,'\n',)
        # print('here',args.attention_threshold)
        #print('here threshold',final_threshold,attention_threshold)
        #print(f"Not preset is used. Using default topK hybrid pruning. (Density: {final_threshold})")
        return SparseTrainingArguments(
            dense_pruning_method="topK:1d_alt",
            attention_pruning_method="topK",
            initial_threshold=1.0,
            layerwise_thresholds=args.layerwise_thresholds,
            initial_warmup=1,
            final_warmup=3,
            attention_block_rows=32,
            attention_block_cols=32,
            attention_output_with_dense=0,
            regularization_final_lambda=20,
            dense_lambda=0.25,
            regularization=None
        )
    print('here')
    with open(args.sparse_preset) as f:
        sparse_params = json.load(f)
        set_dict(sparse_params, args.layerwise_thresholds,
                 'layerwise_thresholds')
        #set_dict(sparse_params, args.attention_threshold, 'attention_threshold')
        return SparseTrainingArguments(**sparse_params)


# def override_sparse_training_arguments_attention_block_size(sparse_training_arguments: SparseTrainingArguments, deit_model_name):
#   print('override sparse traning arguments attention block rows and columns according to attention head sizes')
#   assert 'tiny' in deit_model_name or 'small' in deit_model_name or 'base' in deit_model_name
#   if 'tiny' in deit_model_name:
#     sparse_training_arguments.attention_block_rows = 16
#     sparse_training_arguments.attention_block_cols = 192
#   elif 'small' in deit_model_name:
#     sparse_training_arguments.attention_block_rows = 32
#     sparse_training_arguments.attention_block_cols = 384
#   else:
#     sparse_training_arguments.attention_block_rows = 64
#     sparse_training_arguments.attention_block_cols = 768


def override_args_with_deepspeed(args, deepspeed: Path):
    with open(deepspeed) as f:
        deepspeed_params = json.load(f)
        deepspeed_lr = deepspeed_params['optimizer']['params']['lr']
        deepspeed_micro_batch_size = deepspeed_params['train_micro_batch_size_per_gpu']
        if (args.learning_rate, args.micro_batch_size) != (deepspeed_lr, deepspeed_micro_batch_size):
            print('Info: learning_rate and micro_batch_size mismatch between argparser and deepspeed. Override using deepspeed.')
            args.learning_rate = deepspeed_lr
            args.micro_batch_size = deepspeed_micro_batch_size


def get_optimizer_and_lr_scheduler(args, model, max_steps):
    optimizer = None
    lr_scheduler = None
    if args.optimizer and args.lr_scheduler:
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                params=model.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.AdamW(
                params=model.parameters(), lr=args.learning_rate)
        if args.lr_scheduler == 'cosine':
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=0.1 * max_steps, num_training_steps=max_steps)
        else:
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=0.1 * max_steps, num_training_steps=max_steps)
    return optimizer, lr_scheduler


def main():
    # by default it will take all gpu in system.
    # use env CUDA_VISIBLE_DEVICES to limit gpu number.
    parser = argparse.ArgumentParser()
    # "playground" path is for temporary testing
    parser.add_argument("--output_dir", type=Path,
                        default='./results/playground')
    parser.add_argument("--nn_pruning", action='store_true')
    parser.add_argument("--deit_model_name", type=str,
                        default="facebook/deit-base-patch16-224")
    parser.add_argument("--state_dict", type=Path, default=None)
    parser.add_argument("--pytorch_load", action='store_true',
                        help="Use load_state_dict instead of from_pretrained")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--data_path", type=Path, default='/data/data1/v-xudongwang/imagenet',
                        help='imagenet1k root folder, should contain train and val subdirectory.')
    parser.add_argument('--trainset_length', type=int, default=1281167)
    parser.add_argument("--layerwise_thresholds", type=str, default='-'.join(['h_0.5_d_0.5' for _ in range(12)]),
                        help='The final value of the masking threshold. When using topK, this is the final density. With sigmoied_threshold, a good choice is 0.1')
    parser.add_argument("--deepspeed", type=Path, default=None)
    parser.add_argument("--no_training", action='store_true',
                        help='Save model directly (no fine-tuning)')
    parser.add_argument('--seed', type=int, default=12345)
    # traning_args ----------------------------------
    parser.add_argument('--micro_batch_size', type=int, default=64)
    parser.add_argument('--macro_batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', default=5e-5,
                        type=float, help='learning rate')
    parser.add_argument('--scale_lr', action='store_true',
                        help='linear scale learning rate: lr = lr * batch_size_per_gpu * num_gpus / 512')
    parser.add_argument('--optimizer', default=None,
                        choices=['SGD', 'adamw'], type=str, help='specify the optimizer')
    parser.add_argument('--lr_scheduler', default=None,
                        choices=['cosine'], type=str, help='specify the learning rate scheduler')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=None, help='max training steps, mainly for debug use.')
    parser.add_argument('--sparse_preset', type=str, default=None,
                        help='The JSON file defining the SparseTrainingArguments parameters')
    # finetune_args ---------------------------------
    parser.add_argument('--final_finetune', action='store_true',
                        help='To fill the remaining 0s in dense matrix of pruned model to improve acc')
    parser.add_argument('--finetune_output_name', default='final_finetuned',
                        type=Path, help='Finetuned model output dir name.')
    parser.add_argument('--do_eval', action='store_true',
                        help='evaluate the pruned (or finetune) model')
    # disltil_args ----------------------------------
    parser.add_argument('--do_distil', action='store_true',
                        help='do knowledge distillation only if this argument is set')
    parser.add_argument('--teacher_model', type=str, default='facebook/deit-base-patch16-224',
                        help='The teacher model tot do know distillation.')
    parser.add_argument('--distil_temperature', type=float, default=1.0)
    parser.add_argument('--alpha_distil', type=float, default=1.0,
                        help='loss = alpha_distil * distil_loss + (1 - alpha_distil) * student_loss')

    # Reference:
    # - ImageNet1K has 1281167 training images, 1000 classes
    #
    # - Debug env (validset as trainset):
    # python src/train_main.py --output_dir './results/dummy_mini_pruning_sparsity_50' --nn_pruning
    # python src/train_main.py --output_dir './results/ads_playground' --state_dict './vendor/models/AdsSwiftBERT.bin' --pytorch_load
    # python src/train_main.py --output_dir './results/dummy_mini_pruning_sparsity_80' --nn_pruning --final_threshold 0.2
    # python src/train_main.py --output_dir './results/dummy_mini_pruning_sparsity_90' --nn_pruning --final_threshold 0.1
    #
    # - Real env (Pytorch native):
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_main.py --output_dir './results/default' --trainset_file '../../swiftBertData/data.tsv' --trainset_lcnt 1573820370
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_main.py --output_dir './results/default_pruned_density_50' --trainset_file '../../swiftBertData/data.tsv' --trainset_lcnt 1573820370 --final_threshold 0.5
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_main.py --output_dir './results/default_pruned_density_10' --trainset_file '../../swiftBertData/data.tsv' --trainset_lcnt 1573820370 --final_threshold 0.1
    #
    # - (NOT USE FOR NOW) Deepspeed:
    # 1. run ds_report to check whether the env is ok
    # 2. if CUDA_HOME is undefined, run `export CUDA_HOME=/usr/local/cuda-10.2` first
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 src/train_main.py --output_dir './results/default' --trainset_file '../../swiftBertData/data.tsv' --trainset_lcnt 1573820370 --deepspeed src/deepspeed.json
    # Save AdsSwiftBERT.bin to hf format
    # python src/train_main.py --output_dir './results/AdsSwiftBERT' --state_dict './vendor/models/AdsSwiftBERT.bin' --pytorch_load --no_training
    # 6x512 model
    # python src/train_main.py --output_dir './results/dummy_6x512' --deit_model_name 'facebook/deit-base-patch16-224' --micro_batch_size 1024
    # python src/train_main.py --output_dir './results/dummy_6x512_pruning_sparsity_50' --deit_model_name 'facebook/deit-base-patch16-224' --micro_batch_size 1024 --nn_pruning
    # python src/train_main.py --output_dir './results/dummy_6x512_pruning_sparsity_80' --deit_model_name 'facebook/deit-base-patch16-224' --micro_batch_size 1024 --nn_pruning --final_threshold 0.2
    # python src/train_main.py --output_dir './results/dummy_6x512_pruning_sparsity_90' --deit_model_name 'facebook/deit-base-patch16-224' --micro_batch_size 1024 --nn_pruning --final_threshold 0.1
    #
    # - Using sparse preset
    # python src/train_main.py --sparse_preset topk-hybrid --final_threshold 0.2 --nn_pruning
    # python src/train_main.py --output_dir './results/dummy_mini_pruning_unstructured_sparsity_50' --sparse_preset topk-unstructured --final_threshold 0.5 --nn_pruning
    # python src/train_main.py --output_dir './results/dummy_mini_pruning_unstructured_sparsity_80' --sparse_preset topk-unstructured --final_threshold 0.2 --nn_pruning
    # python src/train_main.py --output_dir './results/dummy_mini_pruning_unstructured_sparsity_90' --sparse_preset topk-unstructured --final_threshold 0.1 --nn_pruning
    # python src/train_main.py --output_dir './results/dummy_6x512_pruning_unstructured_sparsity_50' --deit_model_name 'facebook/deit-base-patch16-224' --micro_batch_size 1024 --sparse_preset topk-unstructured --final_threshold 0.5 --nn_pruning
    # python src/train_main.py --output_dir './results/dummy_6x512_pruning_unstructured_sparsity_80' --deit_model_name 'facebook/deit-base-patch16-224' --micro_batch_size 1024 --sparse_preset topk-unstructured --final_threshold 0.2 --nn_pruning
    # python src/train_main.py --output_dir './results/dummy_6x512_pruning_unstructured_sparsity_90' --deit_model_name 'facebook/deit-base-patch16-224' --micro_batch_size 1024 --sparse_preset topk-unstructured --final_threshold 0.1 --nn_pruning
    #
    # - Magnitude + No training (still broken)
    # python src/train_main.py --no_training --nn_pruning --state_dict './vendor/models/AdsSwiftBERT.bin' --pytorch_load --sparse_preset magnitude-hybrid
    #
    # - XFinal fine-tuning
    # python src/train_main.py --deit_model_name ./results/playground/final --output_dir ./results/playground --final_finetune

    args = parser.parse_args()
    set_random(args.seed)
    if args.final_finetune:
        assert not args.nn_pruning, "final finetune conflicts with pruning"
    # sanity check: GPU available? As later code assumes that there's at least 1 GPU.
    assert torch.cuda.device_count(), "No GPU found!"
    # sanity check: file exists?
    assert os.path.isdir(args.data_path), f"{args.data_path} does not exist!"
    # if args.no_training:
    #   assert args.local_rank == -1, "Do not open distributed training with --no_training"
    if args.sparse_preset is not None:
        if '/' in args.sparse_preset:
            args.sparse_preset = Path(args.sparse_preset)
        else:
            # the script is expected to be run on the project root
            args.sparse_preset = Path("./config") / \
                (args.sparse_preset + ".json")

    # trainer will help us init process group
    # if args.local_rank != -1:
    #   if not args.deepspeed:
    #     dist.init_process_group(backend='nccl')
    #   else:
    #     import deepspeed
    #     deepspeed.init_distributed()

    if args.scale_lr:
        args.learning_rate = args.learning_rate * \
            torch.cuda.device_count() * args.micro_batch_size / 512

    if args.deepspeed:
        override_args_with_deepspeed(args, args.deepspeed)
    if args.macro_batch_size is None:
        if args.local_rank == -1:
            args.macro_batch_size = args.micro_batch_size
        else:
            args.macro_batch_size = args.micro_batch_size * torch.cuda.device_count()

    micro_batch_size = args.micro_batch_size
    macro_batch_size = args.macro_batch_size
    gradient_accumulation_steps = int(
        macro_batch_size / micro_batch_size / torch.cuda.device_count())
    epochs = args.epoch
    max_steps = int(args.trainset_length / macro_batch_size * epochs)
    if args.max_steps:
        max_steps = min(max_steps, args.max_steps)

    is_main = args.local_rank == 0 or args.local_rank == -1

    dist_print(
        is_main, '============================================================')
    dist_print(is_main, "gradient acc steps:", gradient_accumulation_steps)
    dist_print(is_main, "max steps:", max_steps)
    training_args = TrainingArguments(
        output_dir=args.output_dir,          # output directory
        max_steps=max_steps,
        # this param is required even when using IterableDataset
        num_train_epochs=epochs,
        # batch size per device during training
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=64,   # batch size for evaluation
        # number of warmup steps for learning rate scheduler
        warmup_steps=0.1 * max_steps,
        weight_decay=0.01,               # strength of weight decay
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        deepspeed=str(args.deepspeed) if args.deepspeed else None,
        local_rank=args.local_rank,
        save_strategy="no",
        dataloader_num_workers=8
    )

    print(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    dist_print(is_main, 'Start building trainset.')
    trainset, _ = build_dataset(args.data_path, is_train=True, shuffle=True)
    dist_print(is_main, 'Start building set.')
    validset, _ = build_dataset(args.data_path, is_train=False, shuffle=False)

    dist_print(is_main, 'Finish build dataset')
    model = AutoModelForImageClassification.from_pretrained(
        args.deit_model_name)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu", args.local_rank)

    dist_print(is_main, 'Finish build model')
    if args.pytorch_load:
        state_dict = torch.load(args.state_dict)
        swift_converter(state_dict)
        m, u = model.load_state_dict(state_dict, strict=True)
        dist_print(is_main, "Missing:", m)
        dist_print(is_main, "Unexpected: ", u)
        # if args.no_training:
        #   model.save_pretrained(args.output_dir / 'final')
        #   exit(0)

    if args.do_distil:
        teacher_model = AutoModelForImageClassification.from_pretrained(
            args.teacher_model)
        teacher_model.to(device)
        teacher_model.eval()

        distil_args = DistilTrainingArguments(
            teacher_model=teacher_model,
            distil_temperature=args.distil_temperature,
            alpha_distil=args.alpha_distil
        )

    if args.nn_pruning:
        dist_print(is_main, 'Start building mpc')
        # print('before',args)
        sparse_args = sparse_argument_builder(args)
        # override_sparse_training_arguments_attention_block_size(sparse_args, args.deit_model_name)
        # print('sparse',sparse_args)
        # sys.exit()

        mpc = ModelPatchingCoordinator(
            sparse_args=sparse_args,
            model_name_or_path=args.deit_model_name,
            device=device,
            cache_dir=args.output_dir /
            ("checkpoints" + ("" if not args.final_finetune else "_final_finetune")),
            logit_names="logits",
            teacher_constructor=None
        )
       # print(model)
        mpc.patch_model(model)
        #print('patched model',model)
       # sys.exit()

       # sys.exit()
        if args.do_distil:
            trainer = SparserWithTeacherTrainer(
                sparse_args=sparse_args,
                distil_args=distil_args,
                args=training_args,
                model=model,
                train_dataset=trainset,
                eval_dataset=validset,
                optimizers=(args.optimizer, args.lr_scheduler))
        else:
            trainer = SparseWithoutTeacherTrainer(
                sparse_args=sparse_args,
                args=training_args,
                model=model,
                train_dataset=trainset,
                eval_dataset=validset,
                optimizers=(args.optimizer, args.lr_scheduler)
            )

    elif not args.no_training:
        if args.do_distil:
            trainer = TrainerWithTeacher(
                distil_args=distil_args,
                args=training_args,               # training arguments, defined above
                model=model,                      # the instantiated 🤗 Transformers model to be trained
                train_dataset=trainset,           # training dataset
                eval_dataset=validset,            # evaluation dataset
                optimizers=get_optimizer_and_lr_scheduler(
                    args, model, max_steps)
            )
        else:
            trainer = Trainer(
                args=training_args,               # training arguments, defined above
                model=model,                      # the instantiated 🤗 Transformers model to be trained
                train_dataset=trainset,           # training dataset
                eval_dataset=validset,            # evaluation dataset
                optimizers=get_optimizer_and_lr_scheduler(
                    args, model, max_steps)
            )

    if args.final_finetune:
        # unzero in attention head
        unzero_parameters(model, head_only=True)

    if args.nn_pruning:
        trainer.set_patch_coordinator(mpc)

    # if training hangs at "Using /home/$USER/.cache/torch_extensions as PyTorch extensions root..."
    # remove everything in that folder and try again (probably a lock not properly removed)
    if not args.no_training:
        dist_print(is_main, '==== Start training ====')
    #  print('trainer',trainer)
        trainer.train()
    if args.nn_pruning:
        mpc.compile_model(model)
    print('after prune',model)

    if args.local_rank == -1 or args.local_rank == 0:
        # pitfall: Multi processes writing to same file may corrupt saved model!
        save_pretrained_output_dir = args.output_dir / \
            ('final' if not args.final_finetune else args.finetune_output_name)
        model.save_pretrained(save_pretrained_output_dir)
        # torch.save(model, args.output_dir / 'final' / 'checkpoint.pt')

    if args.do_eval:
        dist_print(is_main, '***   Evaluating   *** ')
        eval_dataset, _ = build_dataset(
            args.data_path, is_train=False, shuffle=False, return_dict=False)
        # initial process group
        try:
            model = model.to(device)
            dist.init_process_group(backend='nccl')
        except RuntimeError as e:
            dist_print(is_main, 'Caught error:', e)
            pass

        results = evaluate(eval_dataset, model, 500,
                           distributed=True, num_workers=8)
        dist_print(is_main, f'Evaluating accuracy: {results["eval_accuracy"]}')
        if is_main:
            accuracy_file_name = f'accuracy{int(results["eval_accuracy"] * 10000)}.txt'
            os.system(
                f'touch {save_pretrained_output_dir / accuracy_file_name}')

    dist_print(is_main, '==== Saved Pretrained Model Sparsity ====')
    if is_main:
        show_deit_sparsity(save_pretrained_output_dir)
    dist_print(is_main, 'Work finished, exit successfully.')

    '''
  if args.nn_pruning:
      model = SwiftBERT.from_pretrained(args.output_dir / 'final')
      original_params = model.num_parameters()
      model = optimize_model(model, "dense")
      pruned_params = model.num_parameters()
      print("Original params:", original_params)
      print("After-pruned params:", pruned_params)
      print(model)
  '''


if __name__ == "__main__":
    main()
