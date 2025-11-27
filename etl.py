from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import re
import utils
import config
import auth


def parse_conversation_block(dataset):
    text = dataset['text']
    parts = re.split(r"### (Human|Assistant):\s*", text)

    if len(parts) < 5:
        return {"instruction": None, "response": None}

    instruction = parts[2].strip()
    response = parts[4].strip()
    return {
        'instruction': response,
        'response': instruction
    }


def parse_dataset(ds):
    parsed = ds.map(parse_conversation_block)
    return parsed.remove_columns(["text"])


def tokenize_backtranslation(example, tokenizer, max_length=1024):
    # Format
    input_text = (
        f"### INPUT:\n{example['response']}\n\n"
        f"### TARGET:\n{example['instruction']}"
    )

    # Tokenize full sequence
    tokenized = tokenizer(
        input_text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    # Build labels = copy input_ids
    labels = tokenized["input_ids"].copy()

    # Determine token index where TARGET starts
    target_index = input_text.index("### TARGET:")

    # Get number of tokens before target
    prefix_tokens = tokenizer(
        input_text[:target_index],
        add_special_tokens=False
    )["input_ids"]

    prefix_len = len(prefix_tokens)

    # Mask everything before target tokens
    prefix_len = min(prefix_len, len(labels))
    for i in range(prefix_len):
        labels[i] = -100

    # Mask the padding (Good practice to add)
    # Assuming tokenizer.pad_token_id is the ID for padding
    labels = [
        -100 if token == tokenizer.pad_token_id else label
        for token, label in zip(tokenized["input_ids"], labels)
    ]

    tokenized['labels'] = labels

    return tokenized


def tokenize_dataset(dataset, tokenizer):
    return dataset.map(
        tokenize_backtranslation,
        fn_kwargs={"tokenizer": tokenizer},
        batched=False,
        remove_columns=dataset.column_names
    )


def build_dataset():
    print("Loading raw dataset…")
    raw_dataset = load_dataset(config.DATASET_ID)

    # Get tokenizer
    tokenizer = utils.get_tokenizer()

    print("Parsing…")
    parsed_train_ds = parse_dataset(raw_dataset["train"])
    parsed_test_ds = parse_dataset(raw_dataset["test"])

    print("Tokenizing...")
    tokenized_train = tokenize_dataset(parsed_train_ds, tokenizer)
    tokenized_test = tokenize_dataset(parsed_test_ds, tokenizer)

    print(f"Pushing dataset to HF Hub ({config.HF_DATASET_REPO})…")
    final_dataset = DatasetDict({
        "train": tokenized_train,
        "test": tokenized_test
    })

    login(token=auth.HF_TOKEN)
    final_dataset.push_to_hub(config.HF_DATASET_REPO)

    print("Dataset successfully pushed to Hugging Face Hub!")


if __name__ == "__main__":
    build_dataset()
