import argparse
from pathlib import Path
from transformers import AutoModelForImageClassification
import torch
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=Path, required=True, help='pretrained model path to analyse')
args = parser.parse_args()

model = AutoModelForImageClassification.from_pretrained(args.model_path)
attention = model.vit.encoder.layer[0].attention.attention
qkv_weight = [attention.query.weight, attention.key.weight, attention.value.weight]
qkv_sparsity = [torch.sum(x == 0) / np.prod(x.shape) for x in qkv_weight]
qkv_name = ['query', 'key', 'value']

print('Model Layer0 attention analyse summary')
print('qkv sparsity', qkv_sparsity)

for i in range(3):
    plt.imshow(qkv_weight[i] == 0)
    plt.savefig(args.model_path / f'{qkv_name[i]}_sparsity.png')