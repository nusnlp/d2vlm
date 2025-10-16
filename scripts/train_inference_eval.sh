#!/bin/bash


set -e






########################################################
#### supervised finetuning from pre-trained model (stage-2 of etchat or llama-vid)
####
stage2_path=checkpoints/ETChat-Phi3-Mini-Stage-2
stage3_path="/storage/wenzheng/model_zoo/etchat_showlab/debug"
anno_path="/storage/wenzheng/dataset/D2VLM-Dataset/ET-Instruct/evi.json"
arch_module=d2vlm_arch
infer_anno_dir="/storage/wenzheng/dataset/D2VLM-Dataset/ETBench/evi/"
bash scripts/train_stage_3.sh $stage2_path $stage3_path $anno_path $arch_module
rm -rf $stage3_path/checkpoint-*



########################################################
#### fpo based on the trained model
####
stage2_path=${xxx/sft}  # the model from aforementioned stage, or directly use stage-2 of etchat or llama-vid but may get a bit lower result
stage3_path="/storage/wenzheng/model_zoo/0306_final_important/dpo_focused_with_only_dpo_rvc_1loss_1_opensoucre_try"
anno_path=${xxx/D2VLM-Dataset/ET-Instruct/FPO/efficient_sequence.pt}
arch_module=d2vlm_fpo_arch
infer_anno_dir=/storage/wenzheng/dataset/D2VLM-Dataset/ETBench/evi/
bash scripts/train_fpo.sh $stage2_path $stage3_path $anno_path $arch_module


## inference
bash scripts/inference.sh $stage3_path $infer_anno_dir $arch_module
rm -rf $stage3_path/checkpoint-*
## calculate metrics
python all_benchmark_eval/etbench/new_compute_metrics.py --pred_path "${stage3_path}/etbench"




########################################################
#### fpo with mcqa format for general video understanding (optional)
####
# stage2_path=${xxx/sft}  # the model from aforementioned stage, or directly use stage-2 of etchat or llama-vid but may get a bit lower result
# stage3_path=${xxx/fpo_mcqa}
# anno_path=${xxx/D2VLM-Dataset/ET-Instruct/FPO/efficient_sequence_with_mcqa.pt}
# arch_module=d2vlm_fpo_arch
# infer_anno_dir=${/storage/wenzheng/dataset/D2VLM-Dataset/ETBench/evi/}

# bash scripts/train_fpo.sh $stage2_path $stage3_path $anno_path $arch_module
# bash scripts/inference.sh $stage3_path $infer_anno_dir $arch_module
# rm -rf $stage3_path/checkpoint-*
# python all_benchmark_eval/etbench/new_compute_metrics.py --pred_path "${stage3_path}/etbench"