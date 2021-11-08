#!/bin/bash

TASK=$1
OPTIONS="${@:2}"
function mobile_benchmark_vivo() {
    for model in `ls ${MODEL_DIR}`
    do
        echo "********************" && echo $model &&
        python tools.py mobile_benchmark \
        --model "${MODEL_DIR}/$model" \
        --num_runs=30 \
        --warmup_runs=30 \
        --num_threads=1 \
        --taskset_mask=c0 \
        --serial_number=0000028e2c780e4e \
        --benchmark_binary_dir="/data/local/tmp" \
        --no_root \
        $OPTIONS # --no_xnnpack
    done
}

function mobile_benchmark_pixel1() {
    for model in `ls ${MODEL_DIR}`
    do
        echo "********************" && echo $model &&
        python tools.py mobile_benchmark \
        --model "${MODEL_DIR}/$model" \
        --num_runs=20 \
        --warmup_runs=10 \
        --num_threads=1 \
        --taskset_mask=c \
        --serial_number=FA6A70311471 \
        --benchmark_binary_dir="/data/local/tmp" \
        --no_root \
        $OPTIONS
    done
}

function mobile_benchmark_mi() {
    for model in `ls ${MODEL_DIR}`
    do
        echo "********************" && echo $model &&
        python tools.py mobile_benchmark \
        --model "${MODEL_DIR}/$model" \
        --num_runs=30 \
        --warmup_runs=30 \
        --num_threads=1 \
        --taskset_mask=70 \
        --serial_number=2458c476 \
        --benchmark_binary_dir="/data/local/tmp" \
        --no_root \
        $OPTIONS # --no_xnnpack
    done
}

function mobile_benchmark_pixel4() {
    for model in `ls ${MODEL_DIR}`
    do
        echo "********************" && echo $model &&
        python tools.py mobile_benchmark \
        --model "${MODEL_DIR}/$model" \
        --num_runs=20 \
        --warmup_runs=15 \
        --num_threads=1 \
        --taskset_mask=70
    done
}

function mobile_benchmark_pixel4_thread4() {
    for model in `ls ${MODEL_DIR}`
    do
        echo "********************" && echo $model &&
        python tools.py mobile_benchmark \
        --model "${MODEL_DIR}/$model" \
        --num_runs=30 \
        --warmup_runs=30 \
        --num_threads=4 \
        --taskset_mask=f0
    done
}

function mobile_benchmark_pixel4_thread8() {
    for model in `ls ${MODEL_DIR}`
    do
        echo "********************" && echo $model &&
        python tools.py mobile_benchmark \
        --model "${MODEL_DIR}/$model" \
        --num_runs=30 \
        --warmup_runs=30 \
        --num_threads=8 \
        --taskset_mask=ff
    done
}
$TASK ""
