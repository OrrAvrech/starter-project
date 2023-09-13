import torch
from pathlib import Path
from typing import NamedTuple
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageSequenceClip
from transformers import CLIPProcessor, CLIPModel, XCLIPProcessor, XCLIPModel
from transformers.models.clip.modeling_clip import CLIPOutput
import numpy as np
from action_cls.utils import read_txt, draw_text_on_frames


class CLIPModelZoo(NamedTuple):
    clip_vit_large_14 = "openai/clip-vit-large-patch14"
    clip_vit_base_16 = "openai/clip-vit-base-patch16"


class XCLIPModelZoo(NamedTuple):
    xclip_16_zs = "microsoft/xclip-base-patch16-zero-shot"


def get_clip_outputs(frames: list[np.array], texts: list[str]) -> torch.float32:
    model = CLIPModel.from_pretrained(CLIPModelZoo.clip_vit_base_16)
    processor = CLIPProcessor.from_pretrained(CLIPModelZoo.clip_vit_base_16)

    inputs = processor(text=texts, images=frames,
                       return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logit_scale = model.logit_scale.exp()

    # take the average image embedding over vid frames
    avg_image_embeds = torch.mean(outputs.image_embeds, dim=0).unsqueeze(dim=0)
    text_embeds = outputs.text_embeds

    # cosine similarity as logits
    logits_per_text = torch.matmul(text_embeds, avg_image_embeds.t()) * logit_scale
    logits_per_vid = logits_per_text.t()
    probs = logits_per_vid.softmax(dim=1)
    return probs


def get_xclip_outputs(frames: list[np.array], texts: list[str]) -> torch.float32:
    model = XCLIPModel.from_pretrained(XCLIPModelZoo.xclip_16_zs)
    processor = XCLIPProcessor.from_pretrained(XCLIPModelZoo.xclip_16_zs)

    inputs = processor(text=texts, videos=frames, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    probs = outputs.logits_per_video.softmax(dim=1)
    return probs


def get_clip_probs_on_vid(vid: VideoFileClip, fps: float, texts: list[str]) -> torch.float:
    frames = [frame for frame in vid.iter_frames(fps=fps)]
    outputs, logit_scale = get_clip_outputs(frames=frames, texts=texts)

    # take the average image embedding over vid frames
    avg_image_embeds = torch.mean(outputs.image_embeds, dim=0).unsqueeze(dim=0)
    text_embeds = outputs.text_embeds

    # cosine similarity as logits
    logits_per_text = torch.matmul(text_embeds, avg_image_embeds.t()) * logit_scale
    logits_per_vid = logits_per_text.t()
    probs = logits_per_vid.softmax(dim=1)
    return probs


def main():
    exercises = read_txt(Path("label_map_600.txt"))
    # exercises = ["push ups", "squats", "yoga",
    #              "golf swing", "ballet", "swimming",
    #              "slacklining", "hitting baseball", "strumming guitar", "sidekick"]

    prefix_prompt = "a photo of a person doing"
    texts = [ex for ex in exercises]

    video_path = Path(
        "/Users/orrav/Documents/projects/mia_starter_project/dataset/swimming/video/How_To_Stop_Your_Legs_Sinking_Whilst_Swimming_The_Most_Common_Swim_Mistake.mp4")
    # chunk (subclip) duration [seconds]
    chunk_duration = 2
    # sample x frames per second
    subsample_fps = 16

    clip = VideoFileClip(str(video_path))
    clip = clip.subclip(30, 50)
    t_start = 0
    labeled_videos = []
    while t_start < clip.duration:
        t_end = t_start + chunk_duration
        if t_end >= clip.duration:
            t_end = None
        subclip = clip.subclip(t_start, t_end)
        frames = [frame for frame in subclip.iter_frames(fps=subsample_fps)]

        probs = get_xclip_outputs(frames, texts)
        argmax_prob = int(probs.argmax(dim=1).numpy()[0])
        label = exercises[argmax_prob]
        print(label)
        t_start += chunk_duration

        subclip_labeled = ImageSequenceClip(draw_text_on_frames(subclip, label), fps=subclip.fps)
        labeled_videos.append(subclip_labeled)

    labeled_clip = concatenate_videoclips(labeled_videos)
    labeled_clip.write_videofile("output_video.mp4")


if __name__ == '__main__':
    main()

