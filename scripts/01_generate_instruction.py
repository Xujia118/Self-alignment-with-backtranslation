'''
backwards model is trained to predict instruction with output.
Here the sampled dataset is in normal instruction: output order
So we need to extract the output, at index 1.

Then we use the backwards model to predict instructions.
Some instructions will be of high quality, some will be of low quality.
In the next step, we want to curate the high quality instructions.
Then we train the backward model with curated dataset.
'''


import re
from datasets import load_dataset, Dataset
from transformers import pipeline
from . import utils
import config


# --- Configuration (These should be defined or loaded from a config file) ---
# Replace these with your actual model paths
BASE_MODEL_ID = "base/model/path"
ADAPTER_MODEL_ID = "lora/adapter/path"
DATA_FILE_PATH = "data/responses.json"  # Path to your source data file


# --- Utility Functions for Prompt Formatting and Cleaning ---

def format_prompt(response_text: str) -> str:
    return f"""
    Below is a response that answers to a question. 
    Your task is to write a question that would most appropriately produce the response.

    Response:
    {response_text}

    Question:
    """


def process_generated_text(prompt: str, full_text: str) -> str:
    """
    Cleans the generated text by removing the original prompt, newlines, 
    and common list prefixes (1., -, *, etc.).
    """
    # 1. Remove the original prompt text
    generated = full_text[len(prompt):].strip()

    # 2. Stop at the first newline (to prevent generating multiple questions/sentences)
    generated = generated.split("\n")[0].strip()

    # 3. Remove list-like prefixes (e.g., "1. ", "- ", "* ")
    generated = re.sub(r'^\s*(\d+[\.\)]|\-|\*)\s*', '', generated)

    return generated.strip()


# --- Model Setup Logic (Reusing the pattern we discussed) ---

def setup_model(base_id: str, adapter_id: str):
    model, tokenizer = utils.setup_llm_for_task(
        config.BASE_MODEL_ID, config.ADAPTER_ID, is_training=True)

    # 4. Create Pipeline Generator
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,  # Increased for typical question length
        temperature=0.7,   # Increased for creativity/diversity
        do_sample=True,    # Enable sampling
        batch_size=4,      # Pipeline will handle batching internally based on your loop
    )

    return generator


# --- Main Logic Function for Dataset Generation ---

def generate_instructions_in_batches(dataset: Dataset, generator) -> list:
    """
    Iterates over the input data in batches, generates instructions using the LLM pipeline,
    and cleans the output.

    Args:
        data (list): List of dictionaries, each containing an "output" field.
        generator: The Hugging Face text-generation pipeline object.

    Returns:
        list: A list of generated instruction strings (questions).
    """
    batch_size = config.BATCH_SIZE
    num_samples = len(dataset)
    all_instructions = []

    # Get the column containing the responses (your prompt requires "output")
    response_texts = dataset["output"]

    for i in range(0, num_samples, batch_size):
            batch_indices = range(i, min(i + batch_size, num_samples))
            batch = [response_texts[j] for j in batch_indices]

            # Prepare prompts
            batch_prompts = [format_prompt(response) for response in batch]
            print(f"Processing batch {i // batch_size + 1}/{num_samples // batch_size + 1}...")

            # --- Model Generation ---
            batch_generated = generator(batch_prompts)

            # Clean each generated instruction
            processed_instructions = [
                process_generated_text(prompt, gen[0]["generated_text"])
                for prompt, gen in zip(batch_prompts, batch_generated)
            ]

            all_instructions.extend(processed_instructions)

    return all_instructions


if __name__ == "__main__":
    dataset = load_dataset(config.HF_DATASET_LIMA_SINGLE_TURN, split="train")
    dataset = dataset.select(range(10))

    generator = setup_model(config.BASE_MODEL_ID, config.ADAPTER_ID)

    print("Starting instruction generation...")
    generated_instructions = generate_instructions_in_batches(dataset, generator)
    print("Generation complete.")

    # Check that we generated the correct number of instructions
    if len(generated_instructions) != len(dataset):
        raise ValueError(
            "Generated instructions count mismatch the original dataset count.")

    # Create a new column "instruction" in the dataset
    dataset = dataset.add_column("generated_instruction", generated_instructions)

    dataset.save_to_disk("models/curated_lima")
    dataset.push_to_hub("xujia118/curated_lima_v3")



