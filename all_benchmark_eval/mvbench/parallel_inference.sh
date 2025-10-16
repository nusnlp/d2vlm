#!/bin/bash

set -e

stage3_path=$1
anno_path=$2
arch_module=$3
pred_path=$4


export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export PYTHONPATH="./:$PYTHONPATH"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$((29502 + $IDX))
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} MASTER_PORT=$port python etchat/eval/infer_mvbench.py \
        --anno_path $anno_path \
        --data_path /storage/wenzheng/dataset/D2VLM_other_benchmark/MVBench/video \
        --pred_path $pred_path \
        --model_path $stage3_path \
        --arch_module $arch_module \
        --chunk $CHUNKS \
        --index $IDX &
done

wait


# cd /storage/wenzheng/dataset/etbench

# # 修改这一行，使用正确的路径拼接方式    
# python evaluation/new_compute_metrics.py --pred_path "${stage3_path}/etbench"