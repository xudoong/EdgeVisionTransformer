#!/bin/bash

# try run swift-bert
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 src/train_main.py \
# --trainset_file /data/data4/lzhani/taoky/SwiftTransformers_v1/data/EarlyCrossingValidation-NoAgg.tsv \
# --validset_file /data/data4/lzhani/taoky/SwiftTransformers_v1/data/EarlyCrossingValidation-NoAgg.tsv \
# --macro_batch_size 4096 --nn_pruning

# nn_pruning deit_tiny_plus_eval
function nn_pruning_deit_tiny_plus_eval() {
    python -m torch.distributed.launch --nproc_per_node 1 --master_port 12346 src/train_main.py \
    --deit_model_name facebook/deit-tiny-patch16-224 \
    --sparse_preset topk-hybrid \
    --nn_pruning \
    --final_threshold 0.5 \
    --attention_threshold 0.5 \
    --micro_batch_size 256 \
    --output_dir ./results/playground \
    --epoch 1 \
    --no_training \
    --do_eval
}

# final_finetune first run
function finetune_try() {
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 src/train_main.py \
    --deit_model_name results/playground/deit_tiny_final \
    --sparse_preset topk-hybrid \
    --micro_batch_size 256 \
    --output_dir ./results/playground \
    --epoch 1 \
    --do_eval
}

function debug_deepspeed() {
    python -m torch.distributed.launch --nproc_per_node 1 --master_port 12347 src/train_main.py \
    --output_dir ./results/playground \
    --data_path /data/data1/v-xudongwang/imagenet \
    --deit_model_name results/playground/deit_tiny_final \
    --sparse_preset topk-hybrid-struct-layerwise-tiny \
    --micro_batch_size 64 \
    --epoch 1 \
    --nn_pruning \
    --do_eval \
    --deepspeed src/deepspeed_config/deepspeed_finetune_deit_tiny_sgd_cosinelrs.json
}

function v1_first_run() {
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 src/train_main.py \
    --deit_model_name facebook/deit-tiny-patch16-224 \
    --sparse_preset topk-hybrid-struct-layerwise-tiny \
    --micro_batch_size 256 \
    --epoch 1 \
    --do_eval \
    --nn_pruning
}

function debug_head_pruning() {
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 src/train_main.py \
    --deit_model_name facebook/deit-tiny-patch16-224 \
    --sparse_preset topk-hybrid-struct-layerwise-tiny \
    --layerwise_thresholds h_0.334_d_1.0-h_0.334_d_1.0-h_0.334_d_1.0-h_0.334_d_1.0-h_0.334_d_1.0-h_0.334_d_1.0-h_0.334_d_1.0-h_0.334_d_1.0-h_0.334_d_1.0-h_0.334_d_1.0-h_0.334_d_1.0-h_0.334_d_1.0
    --micro_batch_size 256 \
    --epoch 1 \
    --do_eval \
    --nn_pruning
}

function debug_finetune_sgd_cosine() {
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 src/train_main.py \
    --deit_model_name results/playground/deit_tiny_final \
    --sparse_preset topk-hybrid-struct-layerwise-tiny \
    --micro_batch_size 256 \
    --scale_lr \
    --epoch 1 \
    --do_eval \
    --final_finetune \
    --optimizer SGD \
    --lr_scheduler cosine
}

function debug_nn_pruning_sgd_cosine() {
    python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 src/train_main.py \
    --deit_model_name facebook/deit-tiny-patch16-224 \
    --sparse_preset topk-hybrid-struct-layerwise-tiny \
    --micro_batch_size 256 \
    --epoch 1 \
    --do_eval \
    --nn_pruning \
    --optimizer SGD \
    --lr_scheduler cosine \
    --scale_lr
}

function D1025_submit_layerwise_sgd_adamw_cosine() {
    for py_file in D1024_layerwise_deit_sgd_cosine.py D1024_layerwise_deit_adamw_cosine.py
    do
        cd itp
        python $py_file submit tiny head 2
        python $py_file submit small head 5
        python $py_file submit base head 11
        python $py_file submit tiny dense 90
        python $py_file submit small dense 90
        python $py_file submit base dense 90
        cd -
    done
}

function debug_finetune_kd() {
    python -m torch.distributed.launch --nproc_per_node 4 src/train_main.py \
    --deit_model_name results/deit_small_layerwisethreshold_all_head5_threshold0.834_epoch6/final \
    --micro_batch_size 128 \
    --scale_lr \
    --output_dir results/deit_small_layerwisethreshold_all_head5_threshold0.834_epoch6 \
    --epoch 1 \
    --do_eval \
    --do_distil \
    --final_finetune \
    --finetune_output_name final_finetuned_kd
}

function debug_pruning_kd() {
    python -m torch.distributed.launch --nproc_per_node 4 src/train_main.py \
    --deit_model_name facebook/deit-tiny-patch16-224 \
    --sparse_preset topk-hybrid-struct-layerwise-tiny \
    --micro_batch_size 256 \
    --epoch 1 \
    --do_eval \
    --nn_pruning \
    --do_distil 
}

function D1026_submit_finetune_layerwise_cosine_kd() {
    for py_file in D1026_finetune_layerwise_deit_adamw_cosine_kd.py
    do
        cd itp
        python $py_file submit tiny head 2
        python $py_file submit small head 5
        python $py_file submit base head 11
        python $py_file submit tiny dense 90
        python $py_file submit small dense 90
        python $py_file submit base dense 90
        cd -
    done
}

function D1027_debug_prune_tiny() {
    python -m torch.distributed.launch --nproc_per_node 1 --master_port 12347 src/train_main.py \
    --deit_model_name facebook/deit-tiny-patch16-224 \
    --sparse_preset topk-hybrid-struct-layerwise-tiny \
    --layerwise_thresholds h_0.668_d_0.3-h_0.668_d_0.3-h_0.668_d_0.3-h_0.668_d_0.3-h_0.668_d_0.3-h_0.668_d_0.3-h_0.668_d_0.3-h_0.668_d_0.3-h_0.668_d_0.3-h_0.668_d_0.3-h_0.668_d_0.3-h_0.668_d_0.3 \
    --micro_batch_size 256 \
    --max_steps 10 \
    --nn_pruning
}
function D1027_debug_prune_small() {
    python -m torch.distributed.launch --nproc_per_node 1 --master_port 12347 src/train_main.py \
    --deit_model_name facebook/deit-small-patch16-224 \
    --sparse_preset topk-hybrid-struct-layerwise-small \
    --layerwise_thresholds h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3-h_0.50_d_0.3 \
    --micro_batch_size 128 \
    --max_steps 10 \
    --nn_pruning
}

function D1029_finetune_small_head5_teacher_small() {
    python -m torch.distributed.launch --nproc_per_node 4 src/train_main.py \
    --output_dir results/deit_small_layerwisethreshold_all_head5_threshold0.834_adamw_cosine_kd_alpha0.9_epoch6 \
    --data_path /data/data1/v-xudongwang/imagenet \
    --deit_model_name results/deit_small_layerwisethreshold_all_head5_threshold0.834_adamw_cosine_kd_alpha0.9_epoch6/final \
    --epoch 3 \
    --do_eval \
    --final_finetune \
    --finetune_output_name final_finetuned_adamw_cosine_kd_teacher_deit_small \
    --micro_batch_size 128  --scale_lr \
    --optimizer adamw --lr_scheduler cosine \
    --do_distil --teacher_model facebook/deit-small-patch16-224
}

function D1030_debug_small_head4_ffn90() {
    python -m torch.distributed.launch --nproc_per_node 1 src/train_main.py \
    --data_path /data/data1/v-xudongwang/imagenet \
    --deit_model_name facebook/deit-small-patch16-224 \
    --nn_pruning --sparse_preset topk-hybrid-struct-layerwise-small \
    --layerwise_thresholds h_0.668_d_0.9-h_0.668_d_0.9-h_0.668_d_0.9-h_0.668_d_0.9-h_0.668_d_0.9-h_0.668_d_0.9-h_0.668_d_0.9-h_0.668_d_0.9-h_0.668_d_0.9-h_0.668_d_0.9-h_0.668_d_0.9-h_0.668_d_0.9 \
    --max_steps 16 \
    --do_eval \
    --micro_batch_size 128  --scale_lr \
    --optimizer adamw --lr_scheduler cosine \
    --do_distil --teacher_model facebook/deit-small-patch16-224
}
$1 ""
