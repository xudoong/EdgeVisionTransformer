import torch
import argparse
from nn_pruning.inference_model_patcher import optimize_model

def show(model, skip_embedding=False, skip_layernorm=False, skip_bias=False):
  print("Params:", model.num_parameters())
  zero_param_cnt = 0
  param_numel = 0

  for k, v in model.named_parameters():
    if skip_embedding and "embedding" in k:
      continue
    if skip_layernorm and "LayerNorm" in k:
      continue
    if skip_bias and "bias" in k:
      continue
    zero_mask = v == 0

    with torch.no_grad():
      print(k, float(zero_mask.sum() / zero_mask.numel()), int(zero_mask.sum()), zero_mask.shape, sep='\t')
      zero_param_cnt += zero_mask.sum().item()
      param_numel += zero_mask.numel()

  print("Zero params:", zero_param_cnt)
  #print("Params (for):", param_numel)

if __name__ == "__main__":
  from ..model import SwiftBERT
  parser = argparse.ArgumentParser()
  parser.add_argument("--deit_model_name", type=str)
  parser.add_argument("--nn_pruning", action='store_true')
  parser.add_argument("--skip_embedding", action='store_true')
  parser.add_argument("--skip_layernorm", action='store_true')
  parser.add_argument("--skip_bias", action='store_true')

  args = parser.parse_args()
  # python -m src.inspector.get_sparsity --deit_model_name ./results/playground/final

  model = SwiftBERT.from_pretrained(args.deit_model_name)
  if args.nn_pruning:
    model = optimize_model(model, "dense")
  
  show(model, skip_embedding=args.skip_embedding, skip_layernorm=args.skip_layernorm, skip_bias=args.skip_bias)
