#!/bin/bash

set -e


stage3_path=${1:-"/storage/wenzheng/guazai/temp/v3_feature_focus_enhanced_no_and_with_dpo_10consist_all_small"}



# stage3_path=${1:-"/storage/wenzheng/model_zoo/etchat_hopper_1225/work_dirs/1220_v3_plus_full_tune_masked_v2"}


export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export PYTHONPATH="./:$PYTHONPATH"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$((29502 + $IDX))
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} MASTER_PORT=$port python etchat/eval/infer_etbench_decouple.py \
        --anno_path /storage/wenzheng/dataset/D2VLM_other_benchmark/youcook2/inference_input.json \
        --data_path /storage/wenzheng/dataset/D2VLM_other_benchmark \
        --pred_path $stage3_path/youcook2 \
        --model_path $stage3_path \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose &
done

wait

# 输出格式转化为timechat类型
python all_benchmark_eval/youcook2/convert_result_in_timechat_format.py --pred_path $stage3_path/youcook2

# 测试指标
python all_benchmark_eval/youcook2/eval_in_timechat_format.py --pred_file $stage3_path/youcook2/result_converted.json


