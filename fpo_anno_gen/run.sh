#!/bin/bash

set -e


stage2_path=${1:-"/storage/wenzheng/model_zoo/etchat_showlab/202501_part30/v3_all_input_part30"}
stage3_path="/storage/wenzheng/model_zoo/etchat_showlab/202501_part30/v3_all_input_part30_2ep"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="./:$PYTHONPATH"
deepspeed --include localhost:0 --master_port 60016 fpo_anno_gen/build_fpo_anno.py\
    --model_name_or_path $stage2_path \
    --language_model phi3 \
    --conv_type phi3 \
    --fast_tokenizer True \
    --vision_tower eva_vit \
    --vision_processor clip_center_224 \
    --vision_output_layer -2 \
    --vision_output_token patch \
    --mm_projector qformer \
    --anno_path anno/evi_part30.json \
    --video_path /storage/wenzheng/dataset/ET-Instruct-164K/videos \
    --fps 1 \
    --lora_enable True \
    --lora_lr 5e-5 \
    --tuning_mode attention \
    --use_matching False \
    --use_evidence True  \
    --use_time_tag False \
    --bi_attention True \
    --full_question False \
    --alpha 2.0 \
    --min_video_len 5 \
    --max_video_len 350 \
    --max_num_words 200 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir $stage3_path \
    --save_full_model True \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --fp16 True \
    --consistency_loss True