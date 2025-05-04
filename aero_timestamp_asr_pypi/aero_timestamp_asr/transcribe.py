from typing import Any, Dict, Tuple, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from dtw import dtw
from transformers import PreTrainedModel, ProcessorMixin
from transformers.models.qwen2.modeling_qwen2 import repeat_kv

from .utils import (plot_timestamp_graph, prepare_messages, split_audio,
                    split_tokens_on_spaces)


def get_QKs(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Get the QK matrices from the model.
    Args:
        model (PreTrainedModel): The Aero model.
        inputs (Dict[str, torch.Tensor]): The inputs to the model.
    Returns:
        torch.Tensor: The QK matrices.
    """
    Qs = [None] * model.config.text_config.num_hidden_layers
    Ks = [None] * model.config.text_config.num_hidden_layers
    for i, layer in enumerate(model.language_model.model.layers):
        layer.self_attn.q_proj.register_forward_hook(
            lambda _, input, output, index=i: Qs.__setitem__(index, output[0].detach().clone())
        )
        layer.self_attn.k_proj.register_forward_hook(
            lambda _, input, output, index=i: Ks.__setitem__(index, output[0].detach().clone())
        )

    with torch.inference_mode():
        _ = model(
            **inputs,
        )

    Qs = torch.stack(Qs, dim=0)
    Ks = torch.stack(Ks, dim=0)
    hidden_shape = Qs.shape[:-1]
    head_dim = getattr(
        model.config.text_config,
        "head_dim",
        model.config.text_config.hidden_size // model.config.text_config.num_attention_heads,
    )
    num_key_value_groups = model.config.text_config.num_attention_heads // model.config.text_config.num_key_value_heads
    Qs = Qs.view(*hidden_shape, -1, head_dim).transpose(1, 2)
    Ks = Ks.view(*hidden_shape, -1, head_dim).transpose(1, 2)
    Ks = repeat_kv(Ks, num_key_value_groups)
    QKs = torch.matmul(Qs, Ks.transpose(-1, -2))

    return QKs


def transcribe_with_timestamp(
    audio: Union[str, np.ndarray],
    model: PreTrainedModel,
    processor: ProcessorMixin,
    prompt: str = "Please transcribe the audio into text",
    sampling_rate: int = 16000,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    max_new_tokens: int = 4096,
    plot: bool = False,
    save_path: str = None,
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
    # Because we do pooling, so instead of 2 we times 4
    AUDIO_TIME_PER_TOKEN = processor.audio_processor.hop_length * 4 / processor.audio_processor.sampling_rate
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
    prompt_len = inputs["input_ids"].shape[1]
    transcript = processor.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

    # Another forward to get the attention scores for full sequences

    inputs["attention_mask"] = torch.ones_like(outputs, dtype=torch.bool, device=inputs["input_ids"].device)
    inputs["input_ids"] = outputs

    QKs = get_QKs(model, inputs)

    audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    assert inputs["input_ids"].shape[0] == 1, "Batch Size > 1 is not supported for now. Still working on it ..."
    words, word_tokens = split_tokens_on_spaces(outputs[0, prompt_len:], processor.tokenizer)

    audio_mask = (inputs["input_ids"] == audio_token_id).squeeze(0)
    text_mask = inputs["attention_mask"]
    # Prompt before generation are all false
    text_mask[:, :prompt_len] = 0
    text_mask = text_mask.squeeze(0)

    QKs = QKs[:, :, audio_mask, :][:, :, :, text_mask]
    weights = QKs.softmax(dim=-1)
    matrix = weights.mean(axis=(0, 1))
    matrix = matrix.transpose(0, 1).contiguous()
    alignment = dtw(-matrix.cpu().double().numpy())
    jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
    jump_times = alignment.index2s[jumps] * AUDIO_TIME_PER_TOKEN

    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    begin_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]

    data = [
        dict(word=word, begin=begin, end=end)
        for word, begin, end in zip(words, begin_times, end_times)
        if not word.startswith("<|") and word.strip() not in ".,!?、。"
    ]

    if plot:
        plot_timestamp_graph(
            matrix,
            alignment,
            words,
            word_tokens,
            AUDIO_TIME_PER_TOKEN=AUDIO_TIME_PER_TOKEN,
            save_path=save_path,
        )
        plt.show()

    return transcript, data
