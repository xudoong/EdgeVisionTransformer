# from transformers.convert_graph_to_onnx import convert
from pathlib import Path

# missing tokenizer, so just cannot use convert directly
# convert(framework="pt", model="results/playground/final/", output=Path("results/playground/final/output.onnx"), opset=13)

import torch
import argparse
from model import SwiftBERTOutput

from nn_pruning.inference_model_patcher import optimize_model as nn_optimize

from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
from onnxruntime.quantization import QuantizationMode, quantize, quantize_dynamic, QuantType

def quant(use_original=False):
  if use_original:
    input_model = f'{output_name}.onnx'
    output_model = f'{output_name}-quant.onnx'
  else:
    input_model = f'{output_name}-opt.onnx'
    output_model = f'{output_name}-opt-quant.onnx'
  quantize_dynamic(str(args.model_dir / input_model), 
                   str(args.model_dir / output_model), 
                   weight_type=QuantType.QUInt8,
                  #  optimize_model=False, # onnxruntime 1.8.x requires this arg
                  )

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=Path, default='./results/playground/final')
parser.add_argument("--nn_pruning", action='store_true')
parser.add_argument("--no_opt", action='store_true')
parser.add_argument("--force_opt", action='store_true')
parser.add_argument("--max_ad_length", type=int, default=38)
parser.add_argument("--output_name", type=str, default="output")
parser.add_argument("--opset_version", type=int, default=13)

args = parser.parse_args()
assert not (args.no_opt and args.force_opt), "no_opt and force_opt cannot be set together."
# python src/onnx_export.py --model_dir ./results/dummy_mini/final/

model = SwiftBERTOutput.from_pretrained(args.model_dir)
original_params = model.num_parameters()
model = nn_optimize(model, "dense")
pruned_params = model.num_parameters()
print("Original params:", original_params)
print("After-pruned params:", pruned_params)
print(model)

bert_config = model.config

max_ad_length = args.max_ad_length

print("==== export ====")
output_name = args.output_name
if args.nn_pruning:
  model = nn_optimize(model, "dense")
  print(model)
  output_name += "_removepruned"
torch.onnx.export(
  model,
  (torch.tensor([1] * (max_ad_length)).view(-1, max_ad_length),
    torch.tensor([1] * (max_ad_length)).view(-1, max_ad_length),
    torch.tensor([1] * (max_ad_length)).view(-1, max_ad_length)),
  args.model_dir / f'{output_name}.onnx', 
  input_names=['input_ids', 'attention_mask', 'token_type_ids'],
  output_names=['score'],
  verbose=False,
  export_params=True,
  opset_version=args.opset_version,
  do_constant_folding=True
)

print("==== optimization ====")
if args.nn_pruning or args.no_opt:
  # TODO: how to fix that?
  if not args.force_opt:
    print("No optimization (nn_pruning or set no_opt). Doing quanting only")
    quant(use_original=True)
    exit(0)
  if args.nn_pruning:
    print("Warn: num_heads & hidden_size may be changed during nn_pruning! The optimized result can be incorrect.")
  
optimization_options = BertOptimizationOptions('bert')
optimization_options.embed_layer_norm = True
optimization_options.enable_layer_norm = True
optimization_options.enable_skip_layer_norm = True
optimization_options.enable_bias_gelu = True
optimization_options.enable_attention = True

optimized_model = optimize_model(
    str(args.model_dir / f'{output_name}.onnx'),
    model_type='bert',
    num_heads=bert_config.num_attention_heads,
    hidden_size=bert_config.hidden_size,
    optimization_options=optimization_options)
optimized_model.save_model_to_file(str(args.model_dir / f'{output_name}-opt.onnx'))

print("==== quantize ====")
quant(use_original=True)
quant(use_original=False)
