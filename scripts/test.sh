#!/bin/bash

lora_target=("x_proj" "embeddings" "in_proj" "out_proj")


CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
    --multi_gpu \
    main.py \
    --logger="wandb" \
    --log_every_step 50000 \
    --save_every_step 50000 \
    --model_name="state-spaces/mamba-130m-hf" \
    --lora_rank 8 \
    --lora_target "${lora_target[@]}" \
    --tokenizer_name="state-spaces/mamba-130m-hf" \
    --dataset_name="HAERAE-HUB/KOREAN-WEBTEXT" \
    --dataset_text_field="text" \
    --logger_run_name="mamba-fine-tuning" \
    --output_path="results" \
    --lr 0.002 \
    --epochs 1 \
    --gradient_accumulation_steps 12 \
    --context_len 1024 \
    --train_batch_size 2 \
    --test_batch_size 2 \
    --fill_token="<|endoftext|>"