#!/bin/bash

TASK=$1
OPTIONS="${@:2}"

function distributed_launch() {
    python -m torch.distributed.launch --nproc_per_node 4 ./run_classifier.py \
    --normalize_pruning_by_layer \
    --do_prune \
    --eval_pruned \
    --actually_prune \
    --data_dir /mnt/data/EdgeDL/imagenet2012 \
    --eval_batch_size 500 \
    --at_least_x_heads_per_layer 1 \
    --num_workers 8 \
    --use_huggingface_trainer \
    $OPTIONS
}

function iterative_pruning_base() {
    ./itp/run_itp.sh distributed_launch \
    --deit_type base \
    --prune_number `seq 0 4 132` \
    --exact_pruning \
    --train_batch_size 64 \
    --n_retrain_epochs_after_pruning 3 \
    --retrain_learning_rate 0.000025 \
    --output_dir /mnt/data/EdgeDL/are16heads_results/iterative/base
}

function finetune_many_base() {
    python -m torch.distributed.launch --nproc_per_node 4 finetune_many.py \
    --data_dir /mnt/data/EdgeDL/imagenet2012 \
    --model_path /mnt/data/EdgeDL/are16heads_results/iterative/base \
    --output_dir /mnt/data/EdgeDL/are16heads_results/ \
    --finetune_learning_rate 0.000025 \
    --n_finetune_epochs_after_pruning 3 \
    --finetune_batch_size 64
}
$1 ""