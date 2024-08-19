#!/bin/bash

GPUS=2

source /import/ml-sc-scratch2/shubhangiu/miniconda3/etc/profile.d/conda.sh
NCCL_BLOCKING_WAIT=0  \
torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=25003  llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json  \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path hongfenglu/HungarianDocQA_IT_IFEvalQA EtashGuha/HungarianDocQA_IT_SyntheticQA_sampled_03  \
    --eval_data_path hongfenglu/HungarianDocQA_IT_IFEvalQA \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/japanese_it_3e_tvt_llava \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 30 \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --load_best_model_at_end True \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --metric_for_best_model "loss" \
    --train_vision_tower True \
    --early_stopping True 
    
