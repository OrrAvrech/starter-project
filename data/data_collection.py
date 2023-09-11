from pathlib import Path
from data.utils import scrape_videos, extract_audio, transcribe_speech


def main():
    dataset_dir = Path("./dataset")

    # TODO: scrape videos
    scrape_videos(extractor="ytsearch",
                  prompt="common mistake in push ups",
                  restrict_filenames=True,
                  min_duration=10,
                  max_duration=100,
                  ext="mp4",
                  max_total_duration=60,
                  no_playlist=True)

    # extract audio and transcription from videos
    # for vid_path in dataset_dir.rglob("*.mp4"):
    #     audio_path = extract_audio(vid_path)
    #     transcribe_speech(audio_path)
        # TODO: visualize transcribed text on videos


if __name__ == '__main__':
    main()
