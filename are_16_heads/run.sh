#!/bin/bash

function run_eval () {
    python /data/data1/v-xudongwang/benchmark_tools/are_16_heads/run_classifier.py \
    --task_name "ImageNet1K" \
    --do_eval \
    --do_lower_case \
    $1 \
    --normalize_pruning_by_layer \
    --do_prune \
    --eval_pruned \
    --prune_percent `seq 10 10 90` \
    --actually_prune \
    --data_dir /data/data1/v-xudongwang/imagenet \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --eval_batch_size 32 \
    --output_dir /data/data1/v-xudongwang/models/torch_model 2>&1
}

run_eval ""