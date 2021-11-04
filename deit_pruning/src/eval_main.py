from transformers import Trainer, TrainingArguments, AutoModelForImageClassification
from model import SwiftBERT
from data import AdDataset
import argparse
from pathlib import Path
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from time import perf_counter
import numpy as np
from utils import swift_converter, set_random, build_dataset, evaluate
from trainer import TrainerWithTokenizer

from nn_pruning.inference_model_patcher import optimize_model

def compute_latencies(model, inputs):
  from data import get_token_att_ids
  zero = torch.nn.parameter.Parameter(torch.tensor(0), requires_grad=False)
  one = torch.nn.parameter.Parameter(torch.tensor(1), requires_grad=False)
  latencies = []
  inputs = inputs.copy()
  if inputs.get('labels'):
    del inputs['labels']

  attention_mask, token_type_ids = get_token_att_ids(zero, one, inputs['input_ids'].unsqueeze(0))
  inputs = {key: val.unsqueeze(0) for key, val in inputs.items()}
  inputs['attention_mask'] = attention_mask
  inputs['token_type_ids'] = token_type_ids
  # print(model, inputs)

  # Warmup
  with torch.no_grad():
    model.eval()
    for _ in range(10):
      _ = model(**inputs)

    for _ in range(1000):
      start_time = perf_counter()
      _ = model(**inputs)
      latency = perf_counter() - start_time 
      latencies.append(latency)
  # Compute run statistics
  time_avg_ms = 1000 * np.mean(latencies)
  time_std_ms = 1000 * np.std(latencies) 
  print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}") 
  return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", type=Path, default='./results/playground/final')
  parser.add_argument("--prediction_output", type=Path, default='./results/playground/')
  parser.add_argument("--nn_pruning", action='store_true')

  parser.add_argument("--deit_model_name", type=str, default='facebook/deit-base-patch16-224')
  parser.add_argument("--state_dict", type=Path, default=None)
  parser.add_argument("--pytorch_load", action='store_true', help="Use load_state_dict instead of from_pretrained")
  parser.add_argument('--seed', type=int, default=12345)
  parser.add_argument('--no_latency', action='store_true')
  parser.add_argument("--data_path", type=Path, default='/data/data1/v-xudongwang/imagenet', help='imagenet1k root folder, should contain train and val subdirectory.')
  # python src/eval_main.py --model_dir './results/dummy_mini_pruning_sparsity_50/final' --prediction_output './results/dummy_mini_pruning_sparsity_50/' --nn_pruning
  # python src/eval_main.py --pytorch_load --state_dict './vendor/models/AdsSwiftBERT.bin' --model_dir './results/ads_playground/original' --prediction_output './results/ads_playground/original'
  # python src/eval_main.py --model_dir './results/dummy_mini_pruning_sparsity_90/final' --prediction_output './results/dummy_mini_pruning_sparsity_90/' --nn_pruning
  # python src/eval_main.py --model_dir './results/dummy_mini_pruning_sparsity_80/final' --prediction_output './results/dummy_mini_pruning_sparsity_80/' --nn_pruning
  # python src/eval_main.py --model_dir './results/dummy_6x512/final' --prediction_output './results/dummy_6x512/'
  # python src/eval_main.py --model_dir './results/dummy_6x512_pruning_sparsity_50/final' --prediction_output './results/dummy_6x512_pruning_sparsity_50/' --nn_pruning
  # python src/eval_main.py --model_dir './results/dummy_6x512_pruning_sparsity_80/final' --prediction_output './results/dummy_6x512_pruning_sparsity_80/' --nn_pruning
  # python src/eval_main.py --model_dir './results/dummy_6x512_pruning_sparsity_90/final' --prediction_output './results/dummy_6x512_pruning_sparsity_90/' --nn_pruning

  args = parser.parse_args()
  set_random(args.seed)

  testing_args = TrainingArguments(
    output_dir=args.prediction_output,
    per_device_eval_batch_size=500,
  )
  if args.deit_model_name != 'facebook/deit-base-patch16-224' and not args.pytorch_load:
    print("WARN: --deit_model_name will be only used when --pytorch_load is on. Did you mean --model_dir?")

  if args.pytorch_load:
    raise NotImplementedError('This code path under fixing.')
    model = SwiftBERT.from_pretrained(args.deit_model_name)
    print('before pruning',model)
    state_dict = torch.load(args.state_dict)
    m, u = model.load_state_dict(state_dict, strict=True)
    print("Missing:", m)
    print("Unexpected: ", u)
  else:
    model = AutoModelForImageClassification.from_pretrained(args.model_dir)

  testset, _ = build_dataset(args.data_path, is_train=False, shuffle=False, return_dict=False)

  if args.nn_pruning:
    original_params = model.num_parameters()
    model = optimize_model(model, "dense")
    pruned_params = model.num_parameters()
    print("Original params:", original_params)
    print("After-pruned params:", pruned_params)
  else:
    print("Params:", model.num_parameters())


  print(model)

  model.to('cuda')
  result = evaluate(eval_data=testset, model=model, eval_batch_size=100, distributed=False, num_workers=8)
  print(result)
  exit()
  trainer = Trainer(
    model=model,
    args=testing_args
  )

  results = trainer.predict(testset)
  #print(results.predictions[:, 0:1])

  sigmoid = torch.nn.Sigmoid()

  scores = sigmoid(torch.tensor(results.predictions[:, 0:1]))
  labels = results.label_ids

  # print(labels.shape)
  # print(scores.shape)

  print("AP:", average_precision_score(labels, scores))
  print("AUC:", roc_auc_score(labels, scores))

  # latency
  if not args.no_latency:
    raise NotImplementedError('Measure latency is under fixing.')
    compute_latencies(model, testset[0])

  with open(args.model_dir / 'output.tsv', 'w') as f:
    l = labels.tolist()
    s = scores.tolist()
    # print(l.shape, len(l))
    # print(s.shape, len(s))
    assert len(l) == len(s)

    for label, score in zip(l, s):
      f.write(f"{int(label[0])}\t{score[0]}\n")


if __name__ == "__main__":
  main()
