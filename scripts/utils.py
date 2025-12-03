import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import config


def setup_llm_for_task(
        base_model_id: str, 
        adapter_model_id: str | None = None, 
        is_training: bool=False,
        quantization_config=None
    ):
    """
    Sets up the base model and optionally loads the LoRA adapter for a specified task (training or inference).

    Args:
        base_model_id (str): The identifier for the base LLM (e.g., 'meta-llama/Llama-2-7b-hf').
        adapter_model_id (str): The path or ID to the trained LoRA adapter weights.
        is_training (bool): If True, sets the model to .train() mode. Otherwise, .eval() mode.

    Returns:
        tuple: (model, tokenizer)
    """
    # 1. Base Model Loading
    print(f"Loading base model: {base_model_id}...")
    '''
    dtype is applied at model load time, before quantization or anything else.
    It sets the precision of the actual weights stored in memory.

    But if you quantize, then the compute type is decided by the quantization config,
    not the dtype.
    '''
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        dtype=torch.bfloat16 if quantization_config is None else None,
        quantization_config=quantization_config
    )

    # 2. Tokenizer Loading
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Note: Pad token is often missing/needed for generation and training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # For training, it's often good practice to set a padding side
        if is_training:
            # Recommended for Causal LMs during training/batching
            tokenizer.padding_side = "right"
        else:
            tokenizer.padding_side = "left"  # Recommended for generation efficiency

    # 3. LoRA Adapter Loading
    if adapter_model_id:
        print(f"Loading adapter: {adapter_model_id}...")

        # Define key_mapping if needed (though often unnecessary with standard PEFT saving)
        # Using your original mapping as an example:
        key_mapping = {
            'base_model.model.model.model.model': '',
        }

        model = PeftModel.from_pretrained(
            model,
            adapter_model_id,
            # key_mapping=key_mapping # usually we don't need it
        )

    # 4. Set Model State
    if is_training:
        model.train()
        print("Model set to **TRAIN** mode.")
    else:
        model.eval()
        print("Model set to **EVAL** mode.")

    return model, tokenizer
