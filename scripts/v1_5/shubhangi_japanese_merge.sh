#!/bin/bash

GPUS=1

source /import/ml-sc-scratch2/shubhangiu/miniconda3/etc/profile.d/conda.sh
NCCL_BLOCKING_WAIT=0  \
torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=25001  llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_hf.json \
    --model_name_or_path /import/ml-sc-nlpcheckpoints-scratch2/jonathanl/generic_checkpoints/japanese_llava_v1.5_dareties3 \
    --version v1 \
    --data_path EtashGuha/JapaneseDocQA_IT \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/JapaneseDocQA_IT_llava \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 30 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb
