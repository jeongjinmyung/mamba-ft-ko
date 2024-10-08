#!/bin/bash

lora_target=("x_proj" "embeddings" "in_proj" "out_proj")


CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
    --multi_gpu \
    main.py \
    --logger="wandb" \
    --log_every_step 200 \
    --save_every_step 10000 \
    --model_name="state-spaces/mamba-130m-hf" \
    --lora_rank 8 \
    --lora_target "${lora_target[@]}" \
    --tokenizer_name="state-spaces/mamba-130m-hf" \
    --fill_token="<|endoftext|>" \
    --dataset_name="HAERAE-HUB/KOREAN-WEBTEXT" \
    --dataset_text_field="text" \
    --logger_run_name="mamba-fine-tuning" \
    --output_path="results" \
    --lr 0.002 \
    --epochs 1 \
    --context_len 1024 \
    --gradient_accumulation_steps 8 \
    --train_batch_size 2