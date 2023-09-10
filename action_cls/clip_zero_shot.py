import torch
from pathlib import Path
from typing import List, NamedTuple
from moviepy.editor import VideoFileClip, concatenate_videoclips
from transformers import CLIPProcessor, CLIPModel
from transformers.models.clip.modeling_clip import CLIPOutput
import numpy as np

from action_cls.utils import add_text_overlay, draw_label


class CLIPModelZoo(NamedTuple):
    clip_vit_large_14 = "openai/clip-vit-large-patch14"
    clip_vit_base_16 = "openai/clip-vit-base-patch16"


def get_clip_outputs(images: List[np.array], texts: List[str]) -> [CLIPOutput, torch.float]:
    model = CLIPModel.from_pretrained(CLIPModelZoo.clip_vit_base_16)
    processor = CLIPProcessor.from_pretrained(CLIPModelZoo.clip_vit_base_16)

    inputs = processor(text=texts, images=images,
                       return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logit_scale = model.logit_scale.exp()
    return outputs, logit_scale


def get_clip_probs_on_vid(vid: VideoFileClip, fps: float, texts: List[str]) -> torch.float:
    frames = [frame for frame in vid.iter_frames(fps=fps)]
    outputs, logit_scale = get_clip_outputs(images=frames, texts=texts)

    # take the average image embedding over vid frames
    avg_image_embeds = torch.mean(outputs.image_embeds, dim=0).unsqueeze(dim=0)
    text_embeds = outputs.text_embeds

    # cosine similarity as logits
    logits_per_text = torch.matmul(text_embeds, avg_image_embeds.t()) * logit_scale
    logits_per_vid = logits_per_text.t()
    probs = logits_per_vid.softmax(dim=1)
    return probs


def main():
    exercises = ["squat", "pull up", "basketball shot"]
    prefix_prompt = "a photo of a"
    texts = [prefix_prompt + ex for ex in exercises]

    video_path = Path(
        "/Users/orrav/Documents/projects/mia_starter_project/dataset/basketball_shot/videos/AVOID the Most Common Shooting Error! 🚨  #shorts [v0Q9clOTWg0].mp4")
    # chunk (subclip) duration [seconds]
    chunk_duration = 2
    # sample x frames per second
    subsample_fps = 2

    clip = VideoFileClip(str(video_path))
    clip = clip.subclip(0, 12.2)
    t_start = 0
    labeled_videos = []
    while t_start < clip.duration:
        t_end = t_start + chunk_duration
        if t_end >= clip.duration:
            t_end = None
        subclip = clip.subclip(t_start, t_end)
        probs = get_clip_probs_on_vid(subclip, subsample_fps, texts)
        argmax_prob = int(probs.argmax(dim=1).numpy()[0])
        label = exercises[argmax_prob]
        print(label)
        t_start += chunk_duration
        print(probs)

        subclip_labeled = subclip.fl_image(lambda x: draw_label(x, label))
        labeled_videos.append(subclip_labeled)

    labeled_clip = concatenate_videoclips(labeled_videos)
    labeled_clip.write_videofile("output_video.mp4")


if __name__ == '__main__':
    main()

