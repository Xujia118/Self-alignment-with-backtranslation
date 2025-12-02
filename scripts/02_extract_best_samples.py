import re
from datasets import load_dataset, Dataset
from transformers import pipeline
from . import utils
import config

# 1. We change format_prompt to return a LIST of messages (Chat format)
import re
from datasets import load_dataset, Dataset
from transformers import pipeline
from . import utils
import config


def format_prompt_messages(row):
    return [
        {
            "role": "system",
            "content": """You are an AI Data Curator. Decide if a generated instruction should be kept.

        A good instruction must ask a question or yield an action, and be close to the original instruction, although it could be not as specific or detailed
        Otherwise, discard it.

Answer only in this format:
Reasoning: <brief explanation>
Keep: true/false"""
        },
        {
            "role": "user",
            "content": """
Generated Instruction: "{generated}"
Original Instruction: "{original}"
""".format(generated=row['generated_instruction'], original=row['instruction'])
        }
    ]


def setup_model(base_model_id):
    model, tokenizer = utils.setup_llm_for_task(base_model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.1,    # Keep low to adhere to rules
        do_sample=False,
        batch_size=4,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id
    )
    return generator, tokenizer


def generate_rating_in_batches(dataset: Dataset, generator, tokenizer) -> list:
    batch_size = config.BATCH_SIZE
    num_samples = len(dataset)
    all_decisions = []

    for i in range(0, num_samples, batch_size):
        batch_indices = range(i, min(i + batch_size, num_samples))
        batch = [dataset[j] for j in batch_indices]

        batch_prompts_formatted = []
        for row in batch:
            messages = format_prompt_messages(row)
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            batch_prompts_formatted.append(prompt_str)

        print(
            f"Processing batch {i // batch_size + 1}/{num_samples // batch_size + 1}...")

        batch_ratings = generator(batch_prompts_formatted)

        for r_list in batch_ratings:
            text = r_list[0]['generated_text']


            # Robust extraction
            match = re.search(r"Keep:\s*(true|false)", text, re.IGNORECASE)

            if match:
                decision_str = match.group(1).lower()
                decision_bool = True if decision_str == 'true' else False
                all_decisions.append(decision_bool)
            else:
                # Fallback: If parsing fails, we default to False (Drop) to be safe
                print(f"Parsing Failed on: {text[:30]}...")
                all_decisions.append(False)

    dataset = dataset.add_column("score", all_decisions)
    return dataset


if __name__ == "__main__":
    dataset = load_dataset(config.HF_CURATED_LIMA_V3, split="train")
    generator, tokenizer = setup_model("meta-llama/Llama-3.2-3B-Instruct")

    dataset = dataset.select(range(10))
    dataset = generate_rating_in_batches(dataset, generator, tokenizer)

    # Optional: Quick print to verify
    for row in dataset:
        print(
            f"Gen: {row['generated_instruction'][:40]}... | Score: {row['score']}")


def setup_model(base_model_id):
    model, tokenizer = utils.setup_llm_for_task(base_model_id)

    # Ensure pad token is set for Llama 3
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,  # Give it space to think!
        temperature=0.01,
        do_sample=False,
        batch_size=4,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id
    )
    return generator, tokenizer


def generate_rating_in_batches(dataset: Dataset, generator, tokenizer) -> list:
    batch_size = config.BATCH_SIZE
    num_samples = len(dataset)
    all_decisions = []

    for i in range(0, num_samples, batch_size):
        batch_indices = range(i, min(i + batch_size, num_samples))
        batch = [dataset[j] for j in batch_indices]

        # 2. Convert raw text to Llama 3.2 formatted prompts
        # We use tokenizer.apply_chat_template to add the special tokens
        batch_prompts_formatted = []
        for row in batch:
            messages = format_prompt_messages(row)
            # tokenize=False gives us the formatted string to pass to the pipeline
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            batch_prompts_formatted.append(prompt_str)

        print(
            f"Processing batch {i // batch_size + 1}/{num_samples // batch_size + 1}...")

        batch_ratings = generator(batch_prompts_formatted)

        for r_list in batch_ratings:
            text = r_list[0]['generated_text']

            # Debug: See exactly what Llama 3.2 is saying now
            print(f"DEBUG: {text.replace(chr(10), ' ')}")

            # 3. Robust Extraction
            match = re.search(r"Keep:\s*(true|false)", text, re.IGNORECASE)

            if match:
                decision_str = match.group(1).lower()
                decision_bool = True if decision_str == 'true' else False
                all_decisions.append(decision_bool)
            else:
                # Fallback: If parsing fails, we default to False (Drop) to be safe
                print(f"Parsing Failed on: {text[:30]}...")
                all_decisions.append(False)

    dataset = dataset.add_column("keep", all_decisions)
    return dataset


if __name__ == "__main__":
    dataset = load_dataset(config.HF_CURATED_LIMA_V3, split="train")
    # Need to capture tokenizer now
    generator, tokenizer = setup_model("meta-llama/Llama-3.2-3B-Instruct")

    dataset = generate_rating_in_batches(dataset, generator, tokenizer)
    dataset.save_to_disk("models/best_samples")
