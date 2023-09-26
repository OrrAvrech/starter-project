import torch
import random
import pyrallis
from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass
from moviepy.editor import VideoFileClip, ImageSequenceClip, concatenate_videoclips
from transformers import CLIPProcessor, CLIPModel, XCLIPProcessor, XCLIPModel
import numpy as np
from action_cls.utils import read_txt, draw_text_on_frames, load_yaml

random.seed(0)


@dataclass
class ZeroShotConfig:
    data_config_path: Path
    vid_dir: Path
    num_frames: int
    sample_rate: int
    random_sample: bool
    window_size: float
    output_dir: Path


class CLIPModelZoo(NamedTuple):
    clip_vit_large_14 = "openai/clip-vit-large-patch14"
    clip_vit_base_16 = "openai/clip-vit-base-patch16"


class XCLIPModelZoo(NamedTuple):
    xclip_patch16_32_zs = "microsoft/xclip-base-patch16-zero-shot"
    xclip_patch32_8 = "microsoft/xclip-base-patch32"


def get_clip_outputs(frames: list[np.array], texts: list[str]) -> torch.float32:
    model = CLIPModel.from_pretrained(CLIPModelZoo.clip_vit_base_16)
    processor = CLIPProcessor.from_pretrained(CLIPModelZoo.clip_vit_base_16)

    inputs = processor(text=texts, images=frames, return_tensors="pt", padding=True)
    with torch.no_grad():
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
    model = XCLIPModel.from_pretrained(XCLIPModelZoo.xclip_patch32_8)
    processor = XCLIPProcessor.from_pretrained(XCLIPModelZoo.xclip_patch32_8)

    inputs = processor(text=texts, videos=frames, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_video.softmax(dim=1)
    return probs


def load_trim_vid(vid_path: Path, window_size: float, random_sample: bool):
    vid = VideoFileClip(str(vid_path))
    start_time = 0
    if random_sample and vid.duration > window_size:
        start_time = random.uniform(0, vid.duration - window_size)
    end_time = start_time + window_size
    subclip = vid.subclip(start_time, end_time)
    return subclip


def write_true_pred_vid(vid_path: Path, output_dir: Path, true_pred_videos: list):
    vid_list = list(map(lambda x: x[0], true_pred_videos))
    true_pred_clip = concatenate_videoclips(vid_list)
    output_ex_dir = output_dir / vid_path.parents[1].stem
    output_ex_dir.mkdir(exist_ok=True, parents=True)
    
    first_start = true_pred_videos[0][1]
    last_end = true_pred_videos[-1][2]
    vid_name = f"{vid_path.stem}_{int(first_start)}_{int(last_end)}.mp4"
    true_pred_clip.write_videofile(str(output_ex_dir / vid_name))


def viz_zero_shot_predictions(vid_path: Path, label_map: dict[str, str], cfg: ZeroShotConfig):
    num_frames = cfg.num_frames
    sample_rate = cfg.sample_rate
    output_dir = cfg.output_dir

    vid = load_trim_vid(vid_path, cfg.window_size, cfg.random_sample)
    clip_duration = num_frames * sample_rate / vid.fps

    t_start = 0
    labeled_videos = []
    input_frames = np.zeros((num_frames, *vid.size[::-1], 3))
    while t_start < vid.duration:
        t_end = t_start + clip_duration
        if t_end >= vid.duration:
            t_end = None
        subclip = vid.subclip(t_start, t_end)
        subsample_fps = subclip.fps / sample_rate
        frames = [frame for frame in subclip.iter_frames(fps=subsample_fps)][:num_frames]
        # zero padding
        input_frames[: len(frames), ::] = frames

        text_prompts = list(label_map.keys())
        probs = get_xclip_outputs(list(input_frames), text_prompts)
        argmax_prob = int(probs.argmax(dim=1).numpy()[0])
        max_prob = torch.max(probs)
        max_prompt = text_prompts[argmax_prob]
        t_start += clip_duration

        label = label_map[max_prompt]
        text_to_draw = f"{label}:{max_prob:.2f}"
        print(f"{label}:{max_prob:.2f}")
        subclip_labeled = ImageSequenceClip(
            draw_text_on_frames(subclip, text_to_draw), fps=subclip.fps
        )
        labeled_videos.append(subclip_labeled)
        subclip.close()
    vid.close()

    labeled_clip = concatenate_videoclips(labeled_videos)
    output_ex_dir = output_dir / vid_path.parents[1].stem
    output_ex_dir.mkdir(exist_ok=True, parents=True)
    labeled_clip.write_videofile(str(output_ex_dir / vid_path.name))


def save_true_zero_shot_predictions(vid_path: Path, label_map: dict[str, str], cfg: ZeroShotConfig):
    num_frames = cfg.num_frames
    sample_rate = cfg.sample_rate
    output_dir = cfg.output_dir
    print(str(vid_path))

    vid = load_trim_vid(vid_path, cfg.window_size, cfg.random_sample)
    clip_duration = num_frames * sample_rate / vid.fps
    gt = vid_path.parents[1].stem

    t_start = 0
    true_pred_videos = []
    input_frames = np.zeros((num_frames, *vid.size[::-1], 3))
    while t_start < vid.duration:
        t_end = t_start + clip_duration
        if t_end >= vid.duration:
            t_end = None
        subclip = vid.subclip(t_start, t_end)
        subsample_fps = subclip.fps / sample_rate
        frames = [frame for frame in subclip.iter_frames(fps=subsample_fps)][:num_frames]
        # zero padding
        input_frames[: len(frames), ::] = frames

        text_prompts = list(label_map.keys())
        probs = get_xclip_outputs(list(input_frames), text_prompts)
        argmax_prob = int(probs.argmax(dim=1).numpy()[0])
        max_prompt = text_prompts[argmax_prob]

        pred = label_map[max_prompt]
        print(f"{pred}")
        if pred == gt:
            true_pred_videos.append((subclip, t_start, t_end, pred))
        else:
            if len(true_pred_videos) > 0:
                write_true_pred_vid(vid_path, output_dir, true_pred_videos)
            true_pred_videos = []

        t_start += clip_duration
        subclip.close()
    vid.close()


@pyrallis.wrap()
def main(cfg: ZeroShotConfig):
    exercises = load_yaml(cfg.data_config_path).get("actions", list())
    other_exercises = read_txt(Path("other_exercises.txt"))

    label_map = {}
    for ex in exercises + other_exercises:
        label = ex
        if ex in other_exercises:
            label = "other"
        label_map[ex] = label

    for i, video_path in enumerate(cfg.vid_dir.rglob("*.mp4")):
        try:
            # viz_zero_shot_predictions(video_path, label_map, cfg)
            save_true_zero_shot_predictions(video_path, label_map, cfg)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
