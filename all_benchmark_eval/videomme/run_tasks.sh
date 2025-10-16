#!/bin/bash



set -e




########################################################
####### no_stage_3_evi
####
# stage2_path=checkpoints/ETChat-Phi3-Mini-Stage-2
stage3_path=/storage/wenzheng/model_zoo/D2VLM-Models-HF/D2VLM-Models/d2vlm_mcqa_enhanced
# anno_path=/storage/wenzheng/dataset/ETBench_train_test_anno_HF/train/small/evi_etchat_small.json
arch_module=d2vlm_fpo_arch
infer_anno_dir=/storage/wenzheng/dataset/D2VLM_other_benchmark/Video-MME/videomme/test-00000-of-00001.parquet

# bash scripts/ablations/train_stage_3.sh $stage2_path $stage3_path $anno_path $arch_module
bash all_benchmark_eval/videomme/parallel_inference.sh $stage3_path $infer_anno_dir $arch_module ${stage3_path}/videomme
# python scripts/ablations/convert_result_for_eval.py --pred_path $stage3_path/etbench

# rm -rf $stage3_path/checkpoint-*


#############################################################################################