# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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


import argparse
import collections
import logging
import json
import math
import os
import random
import pickle

import time
import numpy as np
import torch

from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_file', type=Path)
    parser.add_argument('--profile', required=False, action='store_true', help='Enable layer profiling (JSON output)')
    parser.add_argument('--extra', required=False, type=str, default=None)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--seq_len', type=int, default=38)
    # python src/onnx_inference.py --model_file ./results/dummy_mini/final/output.onnx

    args = parser.parse_args()
    print(args)

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    
    profile_arg = "--profile" if args.profile else ""
    extra_arg = args.extra if args.extra else ""

    perf_script = os.path.join("vendor/onnx_scripts", "bert_perf_test.py")

    os.system(f'python {perf_script} {profile_arg} --model "' + str(args.model_file) + f'" --batch_size {args.batch_size} --sequence_length {args.seq_len} --seed {args.seed} {extra_arg}')
    

if __name__ == "__main__":
    main()

