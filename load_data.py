import os
import time
from datasets import load_dataset, load_from_disk


def preprocess(args, tokenizer, accelerator=None):
    # 데이터셋 경로 설정
    dataset_name = args.dataset_name
    data_path = f"./preprocessed/{dataset_name}/data"
    tokenized_train_data_path = (
        f"./preprocessed/{dataset_name}/tokenized_train_datasets"
    )

    # 데이터셋이 이미 존재하면 불러옴
    if os.path.exists(tokenized_train_data_path):
        accelerator.print("Loading preprocessed data")
        tokenized_train_datasets = load_from_disk(tokenized_train_data_path)

        return tokenized_train_datasets

    # hub에서 데이터셋 다운로드. dict 형태로 불러와짐
    if accelerator.is_local_main_process:
        accelerator.print(f"Loading dataset: {dataset_name}")
        raw_datasets = load_dataset(dataset_name)

        raw_datasets.save_to_disk(data_path)
        accelerator.print("Dataset downloaded and saved.")
    else:
        while not os.path.exists(data_path):
            time.sleep(1)
        raw_datasets = load_dataset(data_path)

    accelerator.wait_for_everyone()

    # 데이터셋 토크나이징
    def tokenize_function(example):
        tokenized_example = tokenizer(
            example[args.dataset_text_field],
            truncation=True,
            padding=False,
            max_length=args.context_len,
        )
        return {"input_ids": tokenized_example["input_ids"]}

    if accelerator.is_local_main_process:
        tokenized_train_datasets = raw_datasets["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=os.cpu_count(),
            desc="Running tokenizer on train dataset",
        )
        tokenized_train_datasets.save_to_disk(tokenized_train_data_path)
    else:
        while not os.path.exists(tokenized_train_data_path):
            time.sleep(1)
        tokenized_train_datasets = load_dataset(tokenized_train_data_path)

    accelerator.wait_for_everyone()

    return tokenized_train_datasets
