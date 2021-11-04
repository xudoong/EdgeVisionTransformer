from nn_pruning.inference_model_patcher import optimize_model as nn_optimize
from model import SwiftBERT
from transformers import AutoModelForImageClassification
import sys
model = AutoModelForImageClassification.from_pretrained(sys.argv[1])
# model = SwiftBERTOutput.from_pretrained('results/playground/swift_bert_final')
original_params = model.num_parameters()
print('=== model before optimize ===')
print(model)
model = nn_optimize(model, "dense")
pruned_params = model.num_parameters()
print("Original params:", original_params)
print("After-pruned params:", pruned_params)
print('=== model after optimize ===')
print(model)
