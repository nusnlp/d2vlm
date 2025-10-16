#!/bin/bash

set -e
# 捕获 Ctrl+C 并杀死所有子进程
trap 'kill $(jobs -p) 2>/dev/null' EXIT

stage3_path=$1
anno_path=$2
arch_module=$3




export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export PYTHONPATH="./:$PYTHONPATH"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$((29502 + $IDX))
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} MASTER_PORT=$port python d2vlm/eval/infer_etbench_decouple.py \
        --anno_path $anno_path \
        --data_path /storage/wenzheng/dataset/etbench/videos_compressed \
        --pred_path $stage3_path/etbench \
        --model_path $stage3_path \
        --arch_module $arch_module \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose &
done

wait



# Metric Calculation
# python all_benchmark_eval/etbench/new_compute_metrics.py --pred_path "${stage3_path}/etbench"