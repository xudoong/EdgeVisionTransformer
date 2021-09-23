#!/bin/bash

OPTIONS="${@:1}"
# expected options: --deit_type base|small|tiny  --prune_number `seq 0 132|60|24` --head_importance_file

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

run_eval ""