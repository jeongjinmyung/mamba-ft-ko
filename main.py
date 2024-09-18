import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_model, save_model
import wandb
from tqdm import tqdm
import json
import argparse
import gc

from load_data import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument(
        "--project_name", type=str, default="Mamba2FineTuning", help="project name"
    )
    parser.add_argument("--logger", type=str, default=None, help="logger")
    parser.add_argument("--log_every_step", type=int, default=None, help="print log")
    parser.add_argument("--save_every_step", type=int, default=None, help="model save")
    parser.add_argument("--model_name", type=str, default=None, help="hf hub model")
    parser.add_argument("--lora_rank", type=int, default=8, help="lora rank")
    parser.add_argument("--lora_target", nargs="+", help="lora target modules")
    parser.add_argument(
        "--tokenizer_name", type=str, default=None, help="hf hub tokenizer"
    )
    parser.add_argument("--fill_token", type=str, default=None, help="mask token")
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="hf hub dataset name"
    )
    parser.add_argument(
        "--dataset_text_field", type=str, default=None, help="text field in dataset"
    )
    parser.add_argument(
        "--logger_run_name", type=str, default=None, help="run name for logger"
    )
    parser.add_argument("--output_path", type=str, default=None, help="output dir path")
    parser.add_argument(
        "--test_split_percentage", type=float, default=0.1, help="validation ratio"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation steps in accelerate",
    )
    parser.add_argument(
        "--mixed_precision", type=str, default=None, help="mixed precision"
    )
    parser.add_argument(
        "--context_len", type=int, default=512, help="max length for tokenizer"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="batch size for train"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=8, help="batch size for validation"
    )
    parser.add_argument("--T_0", type=int, default=1000, help="t 0 in scheduler")
    parser.add_argument("--T_mult", type=int, default=1, help="t_mult in scheduler")
    parser.add_argument(
        "--eta_min", type=float, default=0.00005, help="eta_min in scheduler"
    )
    args = parser.parse_args()

    # Initialize the accelerator.
    accelerator = Accelerator(
        log_with=args.logger,
        project_dir=args.output_path,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if accelerator.is_local_main_process:
        if args.output_path is not None:
            os.makedirs(args.output_path, exist_ok=True)
    accelerator.wait_for_everyone()

    set_seed(args.seed)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load peft config
    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=args.lora_target,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # Load Model
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
    )

    merged_model = get_peft_model(pretrained_model, lora_config)
    merged_model.print_trainable_parameters()
    merged_model.to(accelerator.device)

    # Load Dataset
    tokenized_train_datasets, tokenized_test_datasets = preprocess(
        args, tokenizer, accelerator
    )

    # Get the fill token and its id.
    fill_token_id = tokenizer.convert_tokens_to_ids(args.fill_token)

    # Create the data collator.
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    accelerator.print("Preparing DataLoader...")
    train_dataloader = DataLoader(
        tokenized_train_datasets,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    test_dataloader = DataLoader(
        tokenized_test_datasets,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # Estimate the number of steps.
    num_steps = args.epochs * len(tokenized_train_datasets) // args.train_batch_size
    num_steps = num_steps // accelerator.num_processes
    accelerator.print(f"Number of GPU process: {accelerator.num_processes}")
    accelerator.print(f"Estimated number of steps: {num_steps:,}")

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(merged_model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min
    )

    # Prepare model, optimizer, and dataloader for accelerator.
    merged_model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        merged_model, optimizer, train_dataloader, test_dataloader
    )

    # Enable trackers.
    if args.logger is not None:
        accelerator.print(f"Enabling logging for project: {args.logger}")
        experiment_config = vars(args)
        accelerator.init_trackers(
            project_name=args.project_name,
            config=experiment_config,
            init_kwargs={"wandb": {"name": args.logger_run_name}},
        )

    # Training loop.
    step = 0
    running_train_loss = []
    history = {
        "train_loss": [],
        "train_lr": [],
        "test_loss": [],
        "step": [],
    }

    # Ignore tokens during loss calculation.
    ignore_index = -1
    if tokenizer.pad_token is not None:
        ignore_index = tokenizer.pad_token_id

    # Evaluate.
    def evaluate(model, test_dataloader):
        model.eval()
        running_test_loss = []
        for batch in test_dataloader:
            inputs = batch["input_ids"].to(accelerator.device)
            labels = torch.roll(inputs, -1, dims=1)
            labels[:, -1] = fill_token_id

            with torch.no_grad():
                outputs = model(inputs)
            B, C, V = outputs.logits.shape
            outputs = outputs.logits.view(B * C, V)
            targets = labels.view(B * C)
            loss = torch.nn.functional.cross_entropy(
                outputs,
                targets,
                ignore_index=ignore_index,
            )
            running_test_loss.append(loss.item())

        return sum(running_test_loss) / len(running_test_loss)

    average_train_loss = 0.0
    # Set progress bar.
    progress_bar = tqdm(total=num_steps, desc="Training", unit="step", colour="GREEN")
    # Train.
    for epoch in range(args.epochs):
        merged_model.train()
        for batch in train_dataloader:

            inputs = batch["input_ids"].to(accelerator.device)

            # Get the labels by shifting the inputs. Remove the first token. Fill the last token.
            labels = torch.roll(inputs, -1, dims=1)
            labels[:, -1] = fill_token_id

            # Forward
            with accelerator.accumulate(merged_model):

                outputs = merged_model(inputs)
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

            # Next step.
            step += 1

            # Save every step.
            if (
                step % args.save_every_step == 0
                and step > 0
                and args.save_every_step > 0
            ):
                checkpoint_dir = os.path.join(args.output_path, f"checkpoint-{step}")
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    accelerator.unwrap_model(merged_model).save_pretrained(
                        os.path.join(checkpoint_dir, "model_ckp")
                    )
                    save_model(
                        accelerator.unwrap_model(merged_model),
                        os.path.join(checkpoint_dir, "model.safetensors"),
                    )

            # Log every step.
            if (
                step % args.log_every_step == 0
                and step > 0
                and args.log_every_step > 0
                and accelerator.is_local_main_process
            ):

                # evaluate
                average_test_loss = evaluate(merged_model, test_dataloader)
                # Update the log.
                last_lr = lr_scheduler.get_last_lr()[0]
                history["train_loss"].append(average_train_loss)
                history["train_lr"].append(last_lr)
                history["test_loss"].append(average_test_loss)
                history["step"].append(step)
                running_train_loss = []

                # Log to wandb.
                if args.logger is not None:
                    accelerator.log(
                        {
                            "train_loss": average_train_loss,
                            "test_loss": average_test_loss,
                            "lr": last_lr,
                        },
                        step=step,
                    )

                # Update the progressbar. Use the step as the total. Also display the loss and lr.
                progress_bar.set_postfix({"loss": average_train_loss, "lr": last_lr})
                progress_bar.update(args.log_every_step)

    # End training.
    progress_bar.close()
    accelerator.wait_for_everyone()

    # Print some information.
    accelerator.print(f"Training completed. Epochs: {epoch}, Steps: {step}")

    # Save the last model.
    checkpoint_dir = os.path.join(args.output_path, f"checkpoint-{step}-last")
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        accelerator.unwrap_model(merged_model).save_pretrained(
            os.path.join(checkpoint_dir, "model_ckp")
        )
        save_model(
            accelerator.unwrap_model(merged_model),
            os.path.join(checkpoint_dir, "model.safetensors"),
        )

    # Save the history as JSON.
    history_path = os.path.join(args.output_path, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    accelerator.end_training()


if __name__ == "__main__":
    main()
    gc.collect()
    torch.cuda.empty_cache()
