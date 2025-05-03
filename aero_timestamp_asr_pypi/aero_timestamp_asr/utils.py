import string
from typing import Any, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from transformers import PreTrainedTokenizer


def split_audio(audio_arrays, chunk_limit=480000):
    CHUNK_LIM = chunk_limit
    audio_splits = []
    # Split the loaded audio to 30s chunks and extend the messages content
    for i in range(
        0,
        len(audio_arrays),
        CHUNK_LIM,
    ):
        audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
    return audio_splits


def prepare_messages(
    audios: List[Any],
    prompt: str,
) -> List[dict]:
    """
    Prepare messages for the model.
    """

    messages = [{"role": "user", "content": []}]
    for _ in audios:
        messages[0]["content"].append(
            {
                "type": "audio_url",
                "audio": "placeholder",
            }
        )
    messages[0]["content"].append(
        {
            "type": "text",
            "text": prompt,
        }
    )
    return messages


def plot_timestamp_graph(
    matrix: torch.Tensor,
    alignment: Any,
    words: List[str],
    word_tokens: List[List[int]],
    AUDIO_TIME_PER_TOKEN: float = 0.04,
    save_path: str = None,
):
    matrix = matrix.cpu().float().numpy()
    # display the normalized attention weights and the alignment
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, aspect="auto")
    plt.plot(alignment.index2s, alignment.index1s, color="red")

    xticks = np.arange(0, matrix.shape[1], 1 / AUDIO_TIME_PER_TOKEN)
    xticklabels = (xticks * AUDIO_TIME_PER_TOKEN).round().astype(np.int32)
    plt.xticks(xticks, xticklabels)
    plt.xlabel("Time (s)")

    # display tokens and words as tick labels
    ylims = plt.gca().get_ylim()

    ax = plt.gca()
    ax.tick_params("both", length=0, width=0, which="minor", pad=6)

    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_position("left")
    ax.invert_yaxis()
    ax.set_ylim(ylims)

    major_ticks = [-0.5]
    minor_ticks = []
    current_y = 0

    for word, word_token in zip(words, word_tokens):
        minor_ticks.append(current_y + len(word_token) / 2 - 0.5)
        current_y += len(word_token)
        major_ticks.append(current_y - 0.5)

    ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter(words))
    ax.set_yticks(major_ticks)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    plt.ylabel("Words")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)


def split_tokens_on_unicode(tokens: torch.Tensor, tokenizer: PreTrainedTokenizer):
    words = []
    word_tokens = []
    current_tokens = []

    for token in tokens.tolist():
        current_tokens.append(token)
        decoded = tokenizer.decode(current_tokens)
        if "\ufffd" not in decoded:
            words.append(decoded)
            word_tokens.append(current_tokens)
            current_tokens = []

    return words, word_tokens


def split_tokens_on_spaces(tokens: torch.Tensor, tokenizer: PreTrainedTokenizer):
    tokens = filter_special_tokens(tokens, tokenizer)
    subwords, subword_tokens_list = split_tokens_on_unicode(tokens, tokenizer)
    words = [""]
    word_tokens = [[]]
    special_tokens = [tokenizer.encode(tok) for tok in tokenizer.additional_special_tokens]

    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] in special_tokens
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation
        if special or with_space or punctuation:
            words.append(subword)
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)

    return words, word_tokens


def filter_special_tokens(tokens: torch.Tensor, tokenizer: PreTrainedTokenizer):
    special_tokens = [tokenizer.encode(tok)[0] for tok in tokenizer.additional_special_tokens]
    filtered_tokens = []
    for token in tokens:
        if token not in special_tokens:
            filtered_tokens.append(token)
    filtered_tokens = torch.tensor(filtered_tokens, dtype=tokens.dtype)
    return filtered_tokens
