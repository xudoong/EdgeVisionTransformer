import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from logger import logger
import os
import random
import numpy as np
from pathlib import Path

import classifier_training as training
from classifier_scoring import Accuracy
from classifier_eval import evaluate
from util import build_dataset


def add_argument(parser: argparse.ArgumentParser):
    # base args
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--data_dir", default=None, type=str,
                        required=True, help="imagenet 2021 dataset dir")
    parser.add_argument('--model_path', required=True, type=Path,
                        help='Pretrained model dir to perform finetune.')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written."
                        )
    # finetune args
    parser.add_argument("--finetune_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam."
                        )
    parser.add_argument('--scale_learning_rate', action='store_true',
                        help='scale learning rate by lr_scaled = lr * (batch_size_per_gpu * n_gpus) / 512')
    parser.add_argument('--n_finetune_epochs_after_pruning', default=None, type=int,
                        help='Finetune the pruned (or retrained) model for a fixed number of epochs'
                        )
    parser.add_argument('--n_finetune_steps_after_pruning', default=None, type=int,
                        help='Finetune the pruned (or retrained) model for a fixed number of steps'
                        )
    parser.add_argument("--finetune_batch_size", default=64, type=int,
                        help="Finetune batch size per gpu"
                        )
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers to load data'
                        )
    # eval args
    parser.add_argument('--eval_batch_size', default=500, type=int,
                         help='The batch size per gpu to perform evaluation.')


def main():
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()

    # ==== SETUP DEVICE ====
    # This code only support distributed data parallel & gpu training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    logger.info(
        f"device: {device} n_gpu: {n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
    )

    # ==== SETUP EXPERIMENT ====

    def set_seeds(seed, n_gpu):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    set_seeds(42, n_gpu)

    os.makedirs(args.output_dir, exist_ok=True)

    # ==== PREPARE DATA ====
    train_dataset, _ = build_dataset(
        args.data_dir, is_train=True, shuffle=True, return_dict=True)
    eval_dataset, _ = build_dataset(
        args.data_dir, is_train=False, shuffle=False, return_dict=False)

    # ==== PREPARE TRAINING ====
    training_args = training.get_training_args(
        learning_rate=args.finetune_learning_rate,
        micro_batch_size=args.finetune_batch_size,
        n_steps=args.n_finetune_steps_after_pruning,
        n_epochs=args.n_finetune_epochs_after_pruning,
        local_rank=args.local_rank,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    is_main = args.local_rank == -1 or args.local_rank == 0


    def train_func_wrapper(model, output_dir):
        training.huggingface_trainer_train(
            train_dataset=train_dataset,
            model=model,
            args=training_args
        )
        if is_main:
            model.save_pretrained(os.path.join(output_dir, 'final_finetuned'))

    def evaluate_func_wrapper(model, output_dir):
        if is_main:
            logger.info("*** Evaluating ***")
        # Eval accuracy
        scorer = Accuracy()
        accuracy = evaluate(
            eval_dataset,
            model,
            args.eval_batch_size,
            device=device,
            verbose=False,
            scorer=scorer,
            distributed=args.local_rank != -1,
            num_workers=args.num_workers
        )[scorer.name]

        if is_main:
            logger.info("***** Pruning eval results *****")
            logger.info(f"Accuracy\t{accuracy}")
            accuracy_file_name = f'accuracy{int(accuracy * 10000)}.txt'
            os.system(
                f'touch {os.path.join(output_dir, "final_finetuned", accuracy_file_name)}')

    def get_deit_model(model_path):
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained(model_path)
        model.to(device)
        return model


    # === START FINETUNING
    model_list = sorted(os.listdir(args.model_path), key=lambda x: int(x[x.find('prune') + len('prune'): ]))
    i = 0
    while i < len(model_list):
        model_name = model_list[i]
        output_dir = args.model_path / model_name
        # already finetuned
        if 'final_finetuned' in os.listdir(output_dir) and 'config.json' in os.listdir(output_dir / 'final_finetuned'):
            if is_main: logger.info(f'{model_name} already finetuned. Skip. {os.listdir(output_dir / "final_finetuned")}')
            if len(os.listdir(output_dir / 'final_finetuned')) < 3: # not evaluted yet
                model = get_deit_model(output_dir / 'final_finetuned')
                evaluate_func_wrapper(model, output_dir)
        else:
            if is_main: logger.info(f'*** Finetuning {model_name} ***')
            model = get_deit_model(output_dir / 'final')
            train_func_wrapper(model, output_dir)
            evaluate_func_wrapper(model, output_dir)

        if is_main: logger.info('***************************************')
        model_list = sorted(os.listdir(args.model_path), key=lambda x: int(x[x.find('prune') + len('prune'): ]))

        i += 1

    if is_main: logger.info('Done finetune. Exit Successfully.')

if __name__ == '__main__':
    main()
