import argparse
from pathlib import Path
from ..utils import set_random

# import numpy as np
import random

def output(train_filename, output_filename, idx):
  global_idx = 0
  idx_ptr = 0
  fout = open(output_filename, "w")
  with open(train_filename) as f:
    for line in f:
      if global_idx == idx[idx_ptr]:
        fout.write(line)
        idx_ptr += 1
      global_idx += 1
  fout.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_filename", required=True, type=Path)
  parser.add_argument("--output_filename", required=True, type=Path)
  parser.add_argument("--train_lcnt", required=True, type=int),
  parser.add_argument("--ratio", required=True, type=float)
  parser.add_argument("--seed", type=int, default=12345)
  args = parser.parse_args()
  # python -m src.preprocessing.random_select --train_filename ../../swiftBertData/data.tsv --output_filename data/train_subset_0.02.tsv --train_lcnt 1573820370 --ratio 0.02
  ## new dataset: 2000000007
  assert args.train_filename != args.output_filename
  set_random(args.seed)

  selected_lcnt = int(args.train_lcnt * args.ratio)
  print(f"Select {selected_lcnt} / {args.train_lcnt}")

  selected_idx = sorted(random.sample(range(args.train_lcnt), selected_lcnt))
  selected_idx.append(-1)

  output(train_filename=args.train_filename, output_filename=args.output_filename, idx=selected_idx)
