#!/bin/bash



set -e




########################################################
####### no_stage_3_evi
####
# stage2_path=checkpoints/ETChat-Phi3-Mini-Stage-2
stage3_path=/storage/wenzheng/model_zoo/D2VLM-Models-HF/D2VLM-Models/d2vlm_mcqa_enhanced
# anno_path=/storage/wenzheng/dataset/ETBench_train_test_anno_HF/train/small/evi_etchat_small.json
arch_module=d2vlm_fpo_arch
infer_anno_dir=/storage/wenzheng/dataset/D2VLM_other_benchmark/MVBench/json

# bash scripts/ablations/train_stage_3.sh $stage2_path $stage3_path $anno_path $arch_module
bash all_benchmark_eval/mvbench/parallel_inference.sh $stage3_path $infer_anno_dir $arch_module ${stage3_path}/mvbench

python all_benchmark_eval/mvbench/compute_metric_mvbench.py --results_folder ${stage3_path}/mvbench --output_file ${stage3_path}/mvbench/mvbench_metric.txt




# python scripts/ablations/convert_result_for_eval.py --pred_path $stage3_path/etbench

# rm -rf $stage3_path/checkpoint-*
  
# python all_benchmark_eval/etbench/new_compute_metrics.py --pred_path "${stage3_path}/etbench/threshold_abla/040"


#############################################################################################

# "/storage/wenzheng/model_zoo/important_checkpoint_collection_HF/202505_mcqa/with_mcqa_dpo_only_letter/"