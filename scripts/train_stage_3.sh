#!/bin/bash

set -e


stage2_path=$1
stage3_path=$2
anno_path=$3
arch_module=$4


export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="./:$PYTHONPATH"
# export TRITON_CACHE_DIR=/workspace/wenzheng/.triton/autotune
export NCCL_P2P_LEVEL=NVL

torchrun --nproc_per_node 1  --master_port 29666  d2vlm/train/train.py \
    --deepspeed scripts/zero2_bf16.json \
    --model_name_or_path $stage2_path \
    --language_model phi3 \
    --conv_type phi3 \
    --fast_tokenizer True \
    --vision_tower eva_vit \
    --vision_processor clip_center_224 \
    --vision_output_layer -2 \
    --vision_output_token patch \
    --mm_projector qformer \
    --anno_path $anno_path \
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
    --gradient_accumulation_steps 1 \
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
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --bf16 True \
    --consistency_loss True \
    --arch_module $arch_module