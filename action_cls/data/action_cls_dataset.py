import random
import torch
from pathlib import Path
from torch.utils.data import Dataset
from moviepy.editor import VideoFileClip
import numpy as np


class ActionClassificationDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        num_frames: int,
        sample_rate: int,
        random_sampler: bool,
        transform=None,
    ):
        self.data_dir = data_dir
        self.labels = sorted([sub.name for sub in data_dir.iterdir() if sub.is_dir()])
        self.num_classes = len(self.labels)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.labels)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.labels)}
        self.clips = []
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.random_sampler = random_sampler
        self.transform = transform

        for cls in self.labels:
            class_dir = data_dir / cls
            for video_path in class_dir.glob("video/*.mp4"):
                self.clips.append((video_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx) -> tuple[torch.float32, int]:
        video_path, label = self.clips[idx]

        with VideoFileClip(str(video_path)) as vid:
            clip_duration = self.num_frames * self.sample_rate / vid.fps
            subsample_fps = vid.fps / self.sample_rate

            # by default take the start
            # start_time = vid.duration // 2
            start_time = 0
            # for training, random sample
            if self.random_sampler:
                start_time = random.uniform(0, vid.duration - clip_duration)
            end_time = start_time + clip_duration

            video_frames = list(vid.subclip(start_time, end_time).iter_frames(fps=subsample_fps))[
                : self.num_frames
            ]

        video_np = np.moveaxis(np.array(video_frames), [0, -1], [1, 0])
        video_data = {"video": torch.from_numpy(video_np), "audio": None}
        video_data = self.transform(video_data)
        inputs = video_data["video"]
        return inputs, label
