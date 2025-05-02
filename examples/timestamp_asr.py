
from aero_timestamp_asr import load_aero_model, transcribe
import librosa

def load_audio():
    return librosa.load(librosa.ex("libri1"), sr=16000)[0]


def main():
    model, processor = load_aero_model("lmms-lab/Aero-1-Audio", attn_implementation="flash_attention_2")
    audio = load_audio()
    transcript = transcribe(audio, model, processor)

if __name__ == "__main__":
    main()
