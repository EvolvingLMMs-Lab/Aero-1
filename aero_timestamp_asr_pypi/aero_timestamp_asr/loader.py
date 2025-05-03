from typing import Literal, Tuple

from transformers import (AutoModelForCausalLM, AutoProcessor, PreTrainedModel,
                          ProcessorMixin)


def load_aero_model(
    model_name: str,
    attn_implementation: Literal["eager", "flash_attention_2", "sdpa"],
    device_map: str = "cuda",
    torch_dtype: str = "auto",
) -> Tuple[PreTrainedModel, ProcessorMixin]:
    """
    Load the Aero model and processor.

    Args:
        model_name (str): The name of the model to load.
        attn_implementation (str): The attention implementation to use. Can be either "eager" or "flash_attention_2".

    Returns:
        PreTrainedModel: The loaded Aero model.
    """
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, processor
