# Still WIP
from torch.nn.utils import prune
import torch.nn
import argparse
from pathlib import Path
from .block import BlockPruningMethod, block_pruning
from .ln_smart import LnSmartStructured, ln_smart_structured

def is_encoder(name, module):
  return isinstance(module, torch.nn.Linear) and 'bert.encoder' in name

prune_mapping = {
  "random_unstructured": (prune.random_unstructured, prune.RandomUnstructured),
  "l1_unstructured": (prune.l1_unstructured, prune.L1Unstructured),
  "random_structured": (prune.random_structured, prune.RandomStructured),
  "ln_structured": (prune.ln_structured, prune.LnStructured),
  "block": (block_pruning, BlockPruningMethod),
  "ln_smart_structured": (ln_smart_structured, LnSmartStructured),
}

def argbuilder(args):
  if "unstructured" in args.func or args.func == 'block':
    block_args = {}
    if args.func == "block":
      assert args.block_row is not None and args.block_col is not None
      block_args['block_row'] = args.block_row
      block_args['block_col'] = args.block_col
      if args.ln is not None:
        # use fro by default
        block_args['n'] = args.ln
    return {**block_args, **{
      "amount": args.amount
    }}
  else:
    ret = {
      "amount": args.amount,
    }
    if args.func != "ln_smart_structured":
      ret["dim"] = args.dim
    if "ln" in args.func:
      ret = {**ret, **{
        "n": args.ln
      }}
    return ret

def isInt(s):
  try:
    int(s)
    return True
  except ValueError:
    return False

def norm_converter(ln: str):
  if isInt(ln):
    return int(ln)
  elif "inf" in ln:
    return float(ln)
  else:
    return ln

if __name__ == "__main__":
  from ..model import SwiftBERT
  from src.inspector.get_sparsity import show
  from ..utils import set_random

  parser = argparse.ArgumentParser()
  parser.add_argument("--func", type=str)
  parser.add_argument("--global", dest='glob', action='store_true')
  parser.add_argument("--amount", type=float, default=0.5)
  parser.add_argument("--deit_model_name", type=Path, required=True)
  parser.add_argument("--output_dir", type=Path, default='./results/playground/torch_pruned/')
  parser.add_argument("--ln", type=str, default=None)
  parser.add_argument("--dim", type=int, default=None)
  parser.add_argument("--block_row", type=int, default=None)
  parser.add_argument("--block_col", type=int, default=None)
  parser.add_argument("--seed", type=int, default=12345)
  parser.add_argument("--hybrid", action='store_true', help='It overwrites func, global & ln options')
  
  args = parser.parse_args()
  # python -m src.pytorch_prune.pruner --deit_model_name ./results/playground/final
  # python -m src.pytorch_prune.pruner --deit_model_name ./results/AdsSwiftBERT/final/ --func random_unstructured --amount 0.5 --output_dir ./results/AdsSwiftBERT/random_unstructured_0.5

  set_random(args.seed)
  if args.ln is not None:
    args.ln = norm_converter(args.ln)

  model = SwiftBERT.from_pretrained(args.deit_model_name)

  if args.hybrid:
    for name, module in model.named_modules():
      if is_encoder(name, module):
        if "attention" in name:
          block_pruning(module, 'weight', amount=args.amount, block_row=args.block_row, block_col=args.block_col, n='fro')
        elif "dense" in name:
          if args.dim is None:
            ln_smart_structured(module, 'weight', amount=args.amount, n=1)
          else:
            prune.ln_structured(module, 'weight', amount=args.amount, n=1, dim=args.dim)
        else:
          assert 0
        prune.remove(module, 'weight')
  else:
    assert args.func in [
      "random_unstructured",
      "l1_unstructured",
      "random_structured",
      "ln_structured",
      "block",
      "ln_smart_structured"
    ]

    if args.glob:
      assert "_structured" not in args.func

    if "_structured" in args.func:
      if args.func != "ln_smart_structured":
        assert args.dim is not None
      assert not ("ln" in args.func and args.ln is None)

    # start!
    if not args.glob:
      for name, module in model.named_modules():
        if is_encoder(name, module):
          prune_mapping[args.func][0](module, 'weight', **argbuilder(args))
          prune.remove(module, 'weight')
    else:
      parameters_to_prune = []
      for name, module in model.named_modules():
        if is_encoder(name, module):
          parameters_to_prune.append((module, 'weight'))
      prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune_mapping[args.func][1],
        **argbuilder(args)
      )
      for name, module in model.named_modules():
        if is_encoder(name, module):
          prune.remove(module, 'weight')

  # check sparsity
  show(model, skip_embedding=True, skip_layernorm=True, skip_bias=True)

  # export
  model.save_pretrained(args.output_dir)
