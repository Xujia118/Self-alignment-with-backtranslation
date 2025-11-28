import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import config
from utils import get_tokenizer


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    return model


def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


def apply_lora(model):
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj",
                        "fc1", "fc2", "dense", "lm_head"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def train_model():
    # 1️⃣ Load dataset from disk
    dataset = load_dataset(config.HF_DATASET_REPO)

    # 2️⃣ Split
    train_ds = dataset['train']
    test_ds = dataset['test']

    # 3️⃣ Load model and tokenizer
    model = load_model()
    tokenizer = get_tokenizer()
    print_trainable_parameters(model)

    # 4️⃣ Apply LoRA
    model = apply_lora(model)
    print_trainable_parameters(model)

    # 5️⃣ Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        max_steps=config.MAX_STEPS,
        learning_rate=config.LEARNING_RATE,
        
        lr_scheduler_type='cosine',
        optim="paged_adamw_8bit",

        logging_steps=config.LOGGING_STEPS,

        output_dir=config.CHECKPOINT_DIR,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=3,             # Keep last 3 checkpoints

        push_to_hub=True,
        hub_model_id=config.NEW_MODEL_ID,
        hub_strategy="every_save", 

        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        do_eval=True,
        report_to="wandb",
    )

    # 6️⃣ Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 7️⃣ Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=training_args,
        data_collator=data_collator
    )

    # 8️⃣ Train
    trainer.train(resume_from_checkpoint=True)
    trainer.push_to_hub(commit_message="Trained backwards model")
