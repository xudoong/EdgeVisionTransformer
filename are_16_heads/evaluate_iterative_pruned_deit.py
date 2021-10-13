import argparse
import torch
import torch.distributed as dist
from classifier_eval import evaluate
from util import build_dataset
from classifier_scoring import Accuracy
from pathlib import Path
from transformers import AutoModelForImageClassification
import os
from logger import logger

def evaluate_one_model(model_path, dataset, batch_size, local_rank, num_workers):
    is_main = local_rank == -1 or local_rank == 0
    if is_main:
        logger.info(f'*** Evaluating {model_path}***')

    model = AutoModelForImageClassification.from_pretrained(model_path)
    if local_rank == -1:
        device = torch.device('cuda')
    else:
        device = torch.device('cuda', local_rank)
    model.to(device)
    model = (model)
    scorer = Accuracy()
    accuracy = evaluate(
        dataset,
        model,
        batch_size,
        save_attention_probs=False,
        print_head_entropy=False,
        verbose=False,
        scorer=scorer,
        distributed=local_rank != -1,
        num_workers=num_workers
    )[scorer.name]

    if is_main:
        logger.info("***** Pruning eval results *****")
        logger.info(f"Accuracy\t{accuracy}")
        accuracy_file_name = f'accuracy{int(accuracy * 10000)}.txt'
        os.system(f'touch {os.path.join(model_path, accuracy_file_name)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed evaluating')
    parser.add_argument('--data_path', type=Path, default='/data/data1/v-xudongwang/imagenet', help='imagenet2012 dataset path')
    parser.add_argument('--model_path', type=Path, required=True, help='directory of models or a model to evalute')
    parser.add_argument('--batch_size', type=int, default=500, help='evaluate batch size per gpu')
    parser.add_argument('--eval_dir_of_models', action='store_true', help='evaludate all models in model_path')
    parser.add_argument('--num_workers', default=8, type=int, help='dataloader number of workers')
    args = parser.parse_args()

    dataset, _ = build_dataset(args.data_path, is_train=False, shuffle=False, return_dict=False)

    if args.local_rank != -1:
        dist.init_process_group("gloo", rank=args.local_rank, world_size=torch.cuda.device_count())

    if not args.eval_dir_of_models:
        evaluate_one_model(model_path=args.model_path / 'final', dataset=dataset, batch_size=args.batch_size, local_rank=args.local_rank, num_workers=args.num_workers)
    else:
        for model_name in sorted(os.listdir(args.model_path)):
            model_path = args.model_path / model_name
            if 'final' in os.listdir(model_path):
                model_path = model_path / 'final'
                if len(os.listdir(model_path)) < 3:
                    evaluate_one_model(model_path=model_path, dataset=dataset, batch_size=args.batch_size, local_rank=args.local_rank, num_workers=args.num_workers)
                else:
                    logger.info(os.listdir(model_path))
                    logger.info(f"{model_name} already evaluated. Skip.")

if __name__ == '__main__':
    main()