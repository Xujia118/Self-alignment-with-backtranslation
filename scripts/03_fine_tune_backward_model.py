import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, PeftModel
import config
from . import utils


# -----------------------
# LOAD + PREPARE MODEL
# -----------------------
def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model, tokenizer = utils.setup_llm_for_task(
        config.BASE_MODEL_ID,
        config.ADAPTER_ID,
        is_training=True,
        quantization_config=bnb_config
    )
    return model, tokenizer


# -----------------------
# APPLY / LOAD LORA
# -----------------------
def setup_lora(model):
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "fc1", "fc2", "dense", "lm_head"
        ],
        bias="none",
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )

    # load existing adapter
    model = PeftModel.from_pretrained(model, config.ADAPTER_ID)
    model.train()

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    model.print_trainable_parameters()
    return model


# -----------------------
# TOKENIZATION
# -----------------------
def tokenize_fn(example, tokenizer):
    prompt = example["generated_instruction"]
    answer = example["output"]

    full_text = prompt + tokenizer.eos_token + answer + tokenizer.eos_token

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=config.MAX_LENGTH,        # only truncate final combined sequence
    )

    # causal LM: labels match input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def tokenize_dataset(dataset, tokenizer):
    return dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        remove_columns=["output"]
    )


# -----------------------
# TRAINING SETUP
# -----------------------
def build_trainer(model, tokenizer, train_ds, test_ds):
    training_args = TrainingArguments(
        output_dir="new_checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type='cosine',
        optim="paged_adamw_8bit",
        logging_steps=4,
        save_strategy="steps",
        save_steps=4,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=config.ADAPTER_ID,
        hub_strategy="checkpoint",
        eval_strategy="steps",
        eval_steps=4,
        do_eval=True,
        report_to="wandb",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100
        )

    return Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=training_args,
        data_collator=data_collator,
    )


# -----------------------
# MAIN EXECUTION
# -----------------------
def main():
    print("Loading dataset...")
    dataset = load_dataset(config.HF_LIMA_BEST_SAMPLES, split="train")
    split = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_ds, test_ds = split["train"], split["test"]

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    print("Applying LoRA...")
    model = setup_lora(model)

    train_ds = train_ds.select(range(5))
    test_ds = train_ds.select(range(5))

    print("Tokenizing...")
    train_ds = tokenize_dataset(train_ds, tokenizer)
    test_ds = tokenize_dataset(test_ds, tokenizer)

    print("Building trainer...")
    trainer = build_trainer(model, tokenizer, train_ds, test_ds)

    print("Starting training...")
    trainer.train()

    print("Pushing to hub...")
    # trainer.push_to_hub()


if __name__ == "__main__":
    main()
