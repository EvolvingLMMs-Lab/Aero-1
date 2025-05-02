from typing import Union

import librosa
import numpy as np
from transformers import PreTrainedModel, ProcessorMixin

from .utils import prepare_messages, split_audio


def transcribe(
    audio: Union[str, np.ndarray],
    model: PreTrainedModel,
    processor: ProcessorMixin,
    prompt: str = "Please transcribe the audio into text",
    sampling_rate: int = 16000,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    max_new_tokens: int = 4096,
    **kwargs,
):
    if isinstance(audio, str):
        audio = librosa.load(audio, sr=sampling_rate)[0]

    splitted_audio = split_audio(audio)

    messages = prepare_messages(splitted_audio, prompt)

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, audios=splitted_audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
        output_attentions=True,
        return_dict_in_generate=True,
    )

    output_ids = outputs.sequences
    attention_scores = outputs.attentions

    import pdb

    pdb.set_trace()
