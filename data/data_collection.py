from pathlib import Path
from data.utils import extract_audio, transcribe_speech


def main():
    dataset_dir = Path("./dataset")

    # TODO: scrape videos

    # extract audio and transcription from videos
    for vid_path in dataset_dir.rglob("*.mp4"):
        audio_path = extract_audio(vid_path)
        transcribe_speech(audio_path)
        # TODO: visualize transcribed text on videos


if __name__ == '__main__':
    main()
