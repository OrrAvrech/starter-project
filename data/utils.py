import json
from pathlib import Path
from moviepy.editor import VideoFileClip
from transformers import pipeline
from typing import NamedTuple


class ASRModelZoo(NamedTuple):
    whisper_small = "openai/whisper-small"
    wav2vec2 = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


def extract_audio(vid_path: Path, audio_prefix: str = "audio", ext: str = "wav") -> Path:
    with VideoFileClip(str(vid_path)) as clip:
        audio_dir = vid_path.parents[1] / audio_prefix
        audio_dir.mkdir(exist_ok=True)
        filepath = audio_dir / f"{vid_path.stem}.{ext}"
        clip.audio.write_audiofile(filepath)
    return filepath


def transcribe_speech(audio_path: Path, text_prefix: str = "text"):
    # Load pre-trained ASR model
    transcriber = pipeline("automatic-speech-recognition", model=ASRModelZoo.whisper_small)
    transcription = transcriber(str(audio_path))

    text_dir = audio_path.parents[1] / text_prefix
    text_dir.mkdir(exist_ok=True)
    filepath = text_dir / f"{audio_path.stem}.json"
    with open(filepath, "w") as fp:
        json.dump(transcription, fp)

