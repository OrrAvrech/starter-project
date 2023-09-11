import json
import yt_dlp
import subprocess
from pathlib import Path
from moviepy.editor import VideoFileClip
from transformers import pipeline
from typing import NamedTuple


class ASRModelZoo(NamedTuple):
    whisper_small = "openai/whisper-small"
    wav2vec2 = "jonatasgrosman/wav2vec2-large-xlsr-53-english"


def scrape_videos(extractor: str, prompt: str, restrict_filenames: bool,
                  min_duration: int, max_duration: int, ext: str,
                  no_playlist: bool, max_total_duration: int):
    def filter_videos(info_dict):
        duration = info_dict.get('duration')
        if duration and (duration < min_duration or duration > max_duration):
            return "The video is either too short or too long"

    ydl_opts = {"restrictfilenames": restrict_filenames,
                # "match_filter": filter_videos,
                "format": ext,
                "noplaylist": no_playlist}

    max_num_urls = max_total_duration // min_duration
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # error = ydl.download(f"{prompt}")
        search_results = ydl.extract_info(f"{extractor}{max_num_urls}:{prompt}", download=False)

    total_duration = 0
    downloaded_count = 0

    # Loop through the search results and download videos until the maximum duration is reached
    for video_info in search_results['entries']:
        # Check if the video has duration information
        if 'duration' in video_info:
            video_duration = video_info['duration']

            if video_duration < min_duration or video_duration > max_duration:
                continue

            total_duration += video_duration

            # Check if the total aggregated duration exceeds the limit
            if total_duration <= max_total_duration:
                # Get the video URL and title
                video_url = video_info['webpage_url']
                ydl.download([video_url])
                downloaded_count += 1
            else:
                print(f"Maximum aggregated duration of {max_duration} seconds reached.")
                break

    print(f"Downloaded {downloaded_count} videos with a total duration of {total_duration} seconds.")


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

