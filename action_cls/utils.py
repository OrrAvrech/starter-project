import cv2
import yaml
import torch
from pathlib import Path
import numpy as np
from pytorchvideo.transforms import Normalize


class InverseNormalize(Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + torch.finfo(torch.float32).eps)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def draw_label(frame: np.array, text: str):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    rect_height = text_size[1] + 5
    rect_width = text_size[0] + 2

    position = (10, 50)  # (x, y) coordinates
    cv2.rectangle(
        frame,
        (position[0] - 2, position[1] + 5),
        (position[0] + rect_width, position[1] - rect_height),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)


def draw_text_on_frames(vid, text):
    frames = []
    for frame in vid.iter_frames(vid.fps):
        draw_label(frame, text)
        frames.append(frame)
    return frames


def read_txt(filepath: Path) -> list[str]:
    with open(str(filepath), "r") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines


def load_yaml(filepath: Path) -> dict:
    with open(filepath, "r") as fp:
        data = yaml.safe_load(fp)
    return data
