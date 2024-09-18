import os
from datasets import load_dataset, load_from_disk


def preprocess(args, tokenizer, accelerator=None):
    # Set dataset path.
    dataset_name = args.dataset_name
    tokenized_train_data_path = (
        f"./preprocessed/{dataset_name}/tokenized_train_datasets"
    )
    tokenized_test_data_path = f"./preprocessed/{dataset_name}/tokenized_test_datasets"

    # If datasets exist, load them.
    if os.path.exists(tokenized_train_data_path) and os.path.exists(
        tokenized_test_data_path
    ):
        accelerator.print("Loading preprocessed data")
        tokenized_train_datasets = load_from_disk(tokenized_train_data_path)
        tokenized_test_datasets = load_from_disk(tokenized_test_data_path)

        return tokenized_train_datasets, tokenized_test_datasets

    # Download the dataset.
    if accelerator.is_local_main_process:
        accelerator.print(f"Loading dataset: {dataset_name}")
        raw_datasets = load_dataset(dataset_name)
        raw_datasets = raw_datasets["train"].train_test_split(
            test_size=args.test_split_percentage, shuffle=True
        )

    accelerator.wait_for_everyone()

    # Tokenize the datasets.
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

        tokenized_test_datasets = raw_datasets["test"].map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets["test"].column_names,
            num_proc=os.cpu_count(),
            desc="Running tokenizer on eval dataset",
        )
        tokenized_test_datasets.save_to_disk(tokenized_test_data_path)

    accelerator.wait_for_everyone()

    return tokenized_train_datasets, tokenized_test_datasets
