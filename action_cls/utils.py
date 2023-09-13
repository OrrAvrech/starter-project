import cv2
from pathlib import Path
import numpy as np


def draw_label(frame: np.array, text: str):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    font_thickness = 2

    position = (10, 50)  # (x, y) coordinates
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



