#!/bin/bash

TASK=$1
OPTIONS="${@:2}"
# expected options: 
# 1. Eval only: --deit_type base|small|tiny  --prune_number `seq 0 132|60|24` --head_importance_file xxx
# 2. Retrain: --deit_type base|small|tiny  --prune_number `seq 0 132|60|24` --head_importance_file xxx \
#    --train_batch_size 100 --n_retrain_steps_after_pruning  12800 * n_epochs --retrain_learning_rate 1e-3
function run_eval () {
    python /data/data1/v-xudongwang/benchmark_tools/are_16_heads/run_classifier.py \
    --task_name "ImageNet1K" \
    --do_eval \
    --normalize_pruning_by_layer \
    --do_prune \
    --eval_pruned \
    --actually_prune \
    --data_dir /data/data1/v-xudongwang/imagenet \
    --eval_batch_size 500 \
    --at_least_x_heads_per_layer 1 \
    --output_dir /data/data1/v-xudongwang/models/torch_model 2>&1 \
    $OPTIONS
}

function distributed_launch() {
    python -m torch.distributed.launch --nproc_per_node 4 /data/data1/v-xudongwang/benchmark_tools/are_16_heads/run_classifier.py \
    --task_name "ImageNet1K" \
    --do_eval \
    --normalize_pruning_by_layer \
    --do_prune \
    --eval_pruned \
    --actually_prune \
    --data_dir /data/data1/v-xudongwang/imagenet \
    --eval_batch_size 500 \
    --at_least_x_heads_per_layer 1 \
    --output_dir /data/data1/v-xudongwang/models/torch_model 2>&1 \
    $OPTIONS
}

function export_onnx_pruned() {
    python /data/data1/v-xudongwang/benchmark_tools/are_16_heads/run_classifier.py \
    --task_name "ImageNet1K" \
    --normalize_pruning_by_layer \
    --do_prune \
    --actually_prune \
    --data_dir /data/data1/v-xudongwang/imagenet \
    --at_least_x_heads_per_layer 1 \
    --export_onnx \
    --output_dir /data/data1/v-xudongwang/models/torch_model \
    --onnx_output_dir /data/data1/v-xudongwang/models/onnx_model \
    $OPTIONS # --deit_type base --prune_number `seq 0 132` --head_importance_file xxx
}
$TASK ""