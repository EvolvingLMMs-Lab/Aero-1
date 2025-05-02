from typing import Any, List


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
