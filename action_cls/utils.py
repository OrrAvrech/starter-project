import cv2
import numpy as np
from moviepy.editor import TextClip, CompositeVideoClip


def draw_label(frame: np.array, text: str) -> np.array:
    # Specify the font and its characteristics
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    font_thickness = 2

    # Specify the position (bottom-left corner) where you want to place the text
    position = (10, 50)  # (x, y) coordinates

    # Use cv2.putText() to draw the text on the frame
    cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
    return frame


# Create a function to add the text overlay to each frame
def add_text_overlay(video_clip, text):
    # Define the text you want to overlay
    font_size = 40
    font_color = 'white'

    txt_clip = TextClip(txt=text, fontsize=font_size, color=font_color)
    txt_clip = txt_clip.set_position(('right', 'top')).set_duration(video_clip.duration)

    # Overlay the text clip on the first video clip
    video = CompositeVideoClip([video_clip, txt_clip])
    return video
