import json
import yt_dlp
from pathlib import Path
from moviepy.editor import VideoFileClip
from transformers import pipeline
from typing import NamedTuple
from data.data_config import ScraperConfig


class ASRModelZoo(NamedTuple):
    whisper_small = "openai/whisper-small"
    wav2vec2 = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


def scrape_videos(cfg: ScraperConfig, action: str, dataset_dir: Path, video_prefix: str = "video"):
    def filter_videos(info_dict):
        duration = info_dict.get('duration')
        if duration and (duration < cfg.min_vid_duration or duration > cfg.max_vid_duration):
            return "The video is either too short or too long"

    prompt = cfg.prefix_prompt + action
    ydl_opts = {
        "restrictfilenames": cfg.restrict_filenames,
        "match_filter": filter_videos,
        "format": cfg.ext,
        "noplaylist": cfg.no_playlist,
        "quiet": cfg.quiet_mode,
        "outtmpl": {"default": f"{dataset_dir / action / video_prefix}/%(title)s.%(ext)s"},
    }

    agg_duration = cfg.desired_agg_duration
    max_num_urls = agg_duration // cfg.min_vid_duration

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error = ydl.download(f"{cfg.extractor}{max_num_urls}:{prompt}")
        print(error)


def extract_audio(vid_path: Path, audio_prefix: str = "audio", ext: str = "wav") -> Path:
    with VideoFileClip(str(vid_path)) as clip:
        audio_dir = vid_path.parents[1] / audio_prefix
        audio_dir.mkdir(exist_ok=True)
        filepath = audio_dir / f"{vid_path.stem}.{ext}"
        clip.audio.write_audiofile(filepath)
    return filepath


def transcribe_speech(audio_path: Path, chunk_len_s, text_prefix: str = "text"):
    # Load pre-trained ASR model
    transcriber = pipeline("automatic-speech-recognition", model=ASRModelZoo.whisper_small)
    transcription = transcriber(str(audio_path), return_timestamps=True, chunk_length_s=chunk_len_s)

    text_dir = audio_path.parents[1] / text_prefix
    text_dir.mkdir(exist_ok=True)
    filepath = text_dir / f"{audio_path.stem}.json"
    with open(filepath, "w") as fp:
        json.dump(transcription["chunks"], fp)
