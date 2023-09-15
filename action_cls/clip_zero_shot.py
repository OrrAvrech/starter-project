import torch
from pathlib import Path
from typing import NamedTuple
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageSequenceClip
from transformers import CLIPProcessor, CLIPModel, XCLIPProcessor, XCLIPModel
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

    inputs = processor(text=texts, images=frames, return_tensors="pt", padding=True)

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


def viz_zero_shot_predictions(
    vid_path: Path, label_map: dict[str, str], num_frames: int, sample_rate: float, output_dir: Path
):
    vid = VideoFileClip(str(vid_path))
    # vid = vid.subclip(0, 8)
    clip_duration = num_frames * sample_rate / vid.fps
    true_exercise = vid_path.parents[1].name

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
        max_prompt = text_prompts[argmax_prob]
        t_start += clip_duration

        label = label_map[max_prompt]
        print(label)
        subclip_labeled = ImageSequenceClip(draw_text_on_frames(subclip, label), fps=subclip.fps)
        labeled_videos.append(subclip_labeled)

    labeled_clip = concatenate_videoclips(labeled_videos)
    output_dir.mkdir(exist_ok=True)
    labeled_clip.write_videofile(str(output_dir / f"viz_{vid_path.name}"))


def main():
    other_exercises = read_txt(Path("label_map_600.txt"))
    exercises = [
        "push ups",
        "squats",
        "yoga",
        "golf swing",
        "ballet",
        "swimming",
        "slacklining",
        "hitting baseball",
        "strumming guitar",
        "sidekick",
    ]
    label_map = {}
    for ex in exercises + other_exercises:
        label = ex
        if ex in other_exercises:
            label = "other"
        label_map[ex] = label

    video_path = Path(
        "../dataset/yoga/video/1_MISTAKE_Made_by_Yoga_Students_DON_T_Make_This_Mistake.mp4"
    )
    output_dir = Path("./results")
    viz_zero_shot_predictions(
        video_path, label_map, num_frames=32, sample_rate=1, output_dir=output_dir
    )


if __name__ == "__main__":
    main()
