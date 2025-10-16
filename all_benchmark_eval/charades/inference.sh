#!/bin/bash

set -e
# 捕获 Ctrl+C 并杀死所有子进程
trap 'kill $(jobs -p) 2>/dev/null' EXIT

stage3_path=${1:-"/storage/wenzheng/model_zoo/0306_final_important/dpo_focused_with_only_dpo_rvc_1loss_1_opensoucre_try"}



# stage3_path=${1:-"/storage/wenzheng/model_zoo/etchat_hopper_1225/work_dirs/1220_v3_plus_full_tune_masked_v2"}


export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export PYTHONPATH="./:$PYTHONPATH"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$((29502 + $IDX))
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} MASTER_PORT=$port python d2vlm/eval/infer_etbench_decouple.py \
        --anno_path /storage/wenzheng/dataset/D2VLM_other_benchmark/charades/inference_input.json \
        --data_path /storage/wenzheng/dataset/D2VLM_other_benchmark \
        --pred_path $stage3_path/charades \
        --model_path $stage3_path \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose &
done

wait


python all_benchmark_eval/charades/convert_result_in_timechat_format.py --pred_path $stage3_path/charades


python all_benchmark_eval/charades/eval_in_timechat_format.py --pred_file $stage3_path/charades/result_converted.json