from argparse import ArgumentParser
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from logger import logger
import os
import random
import numpy as np

import classifier_args
import classifier_training as training
from classifier_scoring import Accuracy
from classifier_eval import evaluate
from util import build_dataset



def main():
    parser = classifier_args.get_base_parser()
    classifier_args.training_args(parser)
    classifier_args.eval_args(parser)
    classifier_args.finetune_args(parser)

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

    set_seeds(args.seed, n_gpu)

    os.makedirs(args.output_dir, exist_ok=True)

    # ==== PREPARE DATA ====
    train_dataset, _ = build_dataset(args.data_dir, is_train=True, shuffle=True, return_dict=True)
    eval_dataset, _ = build_dataset(args.data_dir, is_train=False, shuffle=False, return_dict=False)

    # ==== PREPARE TRAINING ====
    training_args = training.get_training_args(
        learning_rate=args.finetune_learning_rate,
        micro_batch_size=args.train_batch_size,
        n_steps=args.n_finetune_steps_after_pruning,
        n_epochs=args.n_finetune_epochs_after_pruning,
        local_rank=args.local_rank,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    is_main = args.local_rank == -1 or args.local_rank == 0

    # ==== PREPARE MODEL ====
    def get_deit_model(model_path):
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained(model_path)
        model.to(device)
        return model

    model = get_deit_model(args.finetune_model_path)

    # ==== START TRAINING ====
    training.huggingface_trainer_train(
        train_dataset=train_dataset,
        model=model,
        args=training_args
    )

    if is_main:
        model.save_pretrained(os.path.join(args.output_dir, 'final_finetuned'))

    # ==== EVALUATE ====
    if args.eval_finetuned:
        # Print the pruning descriptor
        if is_main:
            logger.info("*** Evaluating ***")
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
            logger.info(f"Accuracy\t{accuracy}")
            accuracy_file_name = f'accuracy{int(accuracy * 10000)}.txt'
            os.system(f'touch {os.path.join(args.output_dir, "final_finetuned", accuracy_file_name)}')


if __name__ == '__main__':
    main()