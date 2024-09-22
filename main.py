import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_model, save_model
from tqdm import tqdm
import json
import argparse
import gc

from load_data import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--project_name", type=str, default="Mamba2FineTuning", help="project name")
    parser.add_argument("--logger", type=str, default=None, help="logger")
    parser.add_argument("--log_every_step", type=int, default=None, help="print log")
    parser.add_argument("--save_every_step", type=int, default=None, help="model save")
    parser.add_argument("--model_name", type=str, default=None, help="hf hub model")
    parser.add_argument("--lora_rank", type=int, default=8, help="lora rank")
    parser.add_argument("--lora_target", nargs='+', help="lora target modules")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="hf hub tokenizer")
    parser.add_argument("--fill_token", type=str, default=None, help="mask token")
    parser.add_argument("--dataset_name", type=str, default=None, help="hf hub dataset name")
    parser.add_argument("--dataset_text_field", type=str, default=None, help="text field in dataset")
    parser.add_argument("--logger_run_name", type=str, default=None, help="run name for logger")
    parser.add_argument("--output_path", type=str ,default=None, help="output dir path")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps in accelerate")
    parser.add_argument("--mixed_precision", type=str, default=None, help="mixed precision")
    parser.add_argument("--context_len", type=int, default=512, help="max length for tokenizer")
    parser.add_argument("--train_batch_size", type=int, default=8, help="batch size for train")
    parser.add_argument("--T_0", type=int, default=1000, help="t 0 in scheduler")
    parser.add_argument("--T_mult", type=int, default=1, help="t_mult in scheduler")
    parser.add_argument("--eta_min", type=float, default=0.00005, help="eta_min in scheduler")
    args = parser.parse_args()


    # accelerator 초기화
    accelerator = Accelerator(
        log_with=args.logger,
        project_dir=args.output_path,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    # output dir 생성
    if accelerator.is_local_main_process:
        if args.output_path is not None:
            os.makedirs(args.output_path, exist_ok=True)
    accelerator.wait_for_everyone()
    # seed 고정
    set_seed(args.seed)

    # Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # peft config 로드
    lora_config =  LoraConfig(
            r=args.lora_rank,
            target_modules=args.lora_target,
            task_type="CAUSAL_LM",
            bias="none",
    )

    # # 모델 로드. state-space에서는 모델을 float32로 고정할 것을 권장함
    # pretrained_model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name,
    # )

    # # 모델에 lora 적용
    # model = get_peft_model(pretrained_model, lora_config)
    # if accelerator.is_local_main_process:
    #     model.print_trainable_parameters()
    # model.to(accelerator.device)

    # no lora
    model = AutoModelForCausalLM.from_pretrained(
    "OuteAI/Lite-Oute-2-Mamba2Attn-Base",
    # To allow custom modeling files
    trust_remote_code=True,
    )
    model.to(accelerator.device)


    # 데이터셋 로드
    tokenized_train_datasets = preprocess(args, tokenizer, accelerator)


    # fill token 준비
    fill_token_id = tokenizer.convert_tokens_to_ids(args.fill_token)

    # data collator 준비
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # dataloader 준비
    accelerator.print("Preparing DataLoader...")
    train_dataloader = DataLoader(
        tokenized_train_datasets,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,
    )

    # 학습 step 수 설정
    num_steps = args.epochs * len(tokenized_train_datasets) // args.train_batch_size
    num_steps = num_steps // accelerator.num_processes
    accelerator.print(f"Number of GPU process: {accelerator.num_processes}")
    accelerator.print(f"Estimated number of steps: {num_steps:,}")


    # Optimizer, Scheduler 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min
    )

    # accelerator에 model, optimizer, dataloader 준비
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    # trackers 설정
    if args.logger is not None:
        accelerator.print(f"Enabling logging for project: {args.logger}")
        experiment_config = vars(args)
        accelerator.init_trackers(
            project_name=args.project_name, 
            config=experiment_config,
            init_kwargs={"wandb": {"name": args.logger_run_name}}
        )

    # 학습 history
    step = 0
    running_train_loss = []
    history = {
        "loss": [],
        "perplexity": [],
        "lr": [],
        "step": [],
    }


    # fill tokens은 loss 계산 때 무시
    ignore_index = -1
    if tokenizer.pad_token is not None:
        ignore_index = tokenizer.pad_token_id


    average_train_loss = 0.0
    progress_bar = tqdm(total=num_steps, desc="Training", unit="step", colour="GREEN")
    # Train
    model.train()
    for epoch in range(args.epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                inputs = batch['input_ids'].to(accelerator.device)

                # inputs을 shift하여 라벨 설정. 첫 번째 토큰은 fill token 처리
                labels = torch.roll(inputs, -1, dims=1)
                labels[:, -1] = fill_token_id

                outputs = model(inputs)
                B, C, V = outputs.logits.shape
                outputs = outputs.logits.view(B * C, V)
                targets = labels.view(B * C)
                loss = torch.nn.functional.cross_entropy(
                    outputs,
                    targets,
                    ignore_index=ignore_index,
                )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                running_train_loss.append(loss.item())
                average_train_loss = sum(running_train_loss) / len(running_train_loss)
            
            # 다음 step
            step += 1

            # step 별 저장
            if step % args.save_every_step == 0 and step > 0 and args.save_every_step > 0:
                checkpoint_dir = os.path.join(args.output_path, f"checkpoint-{step}")
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    accelerator.unwrap_model(model).save_pretrained(os.path.join(checkpoint_dir, "model_ckp"))
                    save_model(accelerator.unwrap_model(model), os.path.join(checkpoint_dir, "model.safetensors"))
            
            # step 별 로그 확인
            if step % args.log_every_step == 0 and step > 0 and args.log_every_step > 0 and accelerator.is_local_main_process:
                
                try:
                    perplexity = np.exp(average_train_loss)
                except:
                    perplexity = float("-inf")
                
                last_lr = lr_scheduler.get_last_lr()[0]
                history["loss"].append(average_train_loss)
                history["perplexity"].append(perplexity)
                history["lr"].append(last_lr)
                history["step"].append(step)
                running_train_loss = []

                # wandb 로그
                if args.logger is not None:
                    accelerator.log({"loss": average_train_loss, "perplexity": perplexity, "lr": last_lr}, step=step)

                # progress bar 업데이트
                progress_bar.set_postfix({"loss": average_train_loss, "lr": last_lr})
                progress_bar.update(args.log_every_step)

    # training 종료
    progress_bar.close()
    accelerator.wait_for_everyone()
    accelerator.print(f"Training completed. Epochs: {epoch}, Steps: {step}")

    # 마지막 모델 저장
    checkpoint_dir = os.path.join(args.output_path, f"checkpoint-{step}-last")
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        accelerator.unwrap_model(model).save_pretrained(os.path.join(checkpoint_dir, "model_ckp"))
        save_model(accelerator.unwrap_model(model), os.path.join(checkpoint_dir, "model.safetensors"))

    # history 저장
    history_path = os.path.join(args.output_path, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    accelerator.end_training()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()