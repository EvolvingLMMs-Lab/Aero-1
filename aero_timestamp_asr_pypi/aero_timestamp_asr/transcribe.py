from typing import Any, Dict, Tuple, Union

import librosa
import numpy as np
import torch
from transformers import PreTrainedModel, ProcessorMixin

from .monkey_patch import apply_flash_attn_with_attn_scores
from .utils import prepare_messages, split_audio


def transcribe_with_timestamp(
    audio: Union[str, np.ndarray],
    model: PreTrainedModel,
    processor: ProcessorMixin,
    prompt: str = "Please transcribe the audio into text",
    sampling_rate: int = 16000,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    max_new_tokens: int = 4096,
    **kwargs,
) -> Tuple[str, torch.Tensor, Dict[str, Any]]:
    """
    Transcribe the audio using the Aero model.
    Args:
        audio (Union[str, np.ndarray]): The audio file path or numpy array.
        model (PreTrainedModel): The Aero model.
        processor (ProcessorMixin): The Aero processor.
        prompt (str): The prompt to use for transcription.
        sampling_rate (int): The sampling rate of the audio.
        eos_token_id (int): The end-of-sequence token ID.
        pad_token_id (int): The padding token ID.
        max_new_tokens (int): The maximum number of new tokens to generate.
        **kwargs: Additional arguments for the model.
    Returns:
        Tuple[str, torch.Tensor, Dict[Any]]: The transcript, model outputs, and inputs.
    """
    if isinstance(audio, str):
        audio = librosa.load(audio, sr=sampling_rate)[0]

    splitted_audio = split_audio(audio)

    messages = prepare_messages(splitted_audio, prompt)

    processor.tokenizer.padding_side = "left"
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, audios=splitted_audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
    )
    transcript = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    if model.config._attn_implementation == "flash_attention_2":
        apply_flash_attn_with_attn_scores(model, model.config.model_type)

    # Another forward to get the attention scores for full sequences

    inputs["attention_mask"] = torch.ones_like(outputs, dtype=torch.bool, device=inputs["input_ids"].device)
    inputs["input_ids"] = outputs

    with torch.inference_mode():
        forward_output = model(
            **inputs,
            output_attentions=True,
        )

    audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    audio_mask = inputs["input_ids"] == audio_token_id
    audio_seq_len = audio_mask.sum(dim=-1)
    attention_weights = forward_output.attentions
    selected_weights = []
    for weight in attention_weights:
        audio_tokens_attentions = weight[audio_mask]
        selected_weights.append(weight[audio_mask])

    import pdb

    pdb.set_trace()

    return transcript, outputs, inputs
