import pyrallis
from pathlib import Path
from data.utils import scrape_videos, extract_audio, transcribe_speech
from data.data_config import DataConfig


@pyrallis.wrap()
def main(cfg: DataConfig):
    dataset_dir = Path("./dataset")

    actions = cfg.actions
    for action in actions:
        print(f"{action}:")
        scrape_videos(cfg=cfg.scraper, action=action, dataset_dir=dataset_dir)

    # extract audio and transcription from videos
    for vid_path in dataset_dir.rglob("*.mp4"):
        audio_path = extract_audio(vid_path)
        transcribe_speech(audio_path)
        # TODO: visualize transcribed text on videos


if __name__ == '__main__':
    main()
