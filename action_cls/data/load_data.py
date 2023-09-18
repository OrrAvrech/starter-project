from pathlib import Path
from moviepy.editor import ImageSequenceClip

import torch
from torch.utils.data import DataLoader

from action_cls.data.action_cls_dataset import ActionClassificationDataset
from action_cls.utils import InverseNormalize
from action_cls.models.x3d import LitX3DTransfer
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale
)
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, Resize
from torchvision.transforms._transforms_video import CenterCropVideo
import pytorch_lightning as pl


def main():
    data_dir = Path("dataset")
    model_name = "x3d_xs"
    num_classes = 2

    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    frames_per_second = 30
    model_transform_params = {
        "x3d_xs": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "x3d_m": {
            "side_size": 256,
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 5,
        }
    }

    transform_params = model_transform_params[model_name]

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        ShortSideScale(size=transform_params["side_size"]),
                        CenterCropVideo(crop_size=(transform_params["crop_size"], transform_params["crop_size"])),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        ShortSideScale(size=transform_params["side_size"]),
                        CenterCropVideo(crop_size=(transform_params["crop_size"], transform_params["crop_size"]))
                    ]
                ),
            ),
        ]
    )

    train_ds = ActionClassificationDataset(
        data_dir=data_dir,
        num_frames=transform_params["num_frames"],
        sample_rate=transform_params["sampling_rate"],
        random_sampler=True,
        transform=train_transform,
    )

    val_ds = ActionClassificationDataset(
        data_dir=data_dir,
        num_frames=transform_params["num_frames"],
        sample_rate=transform_params["sampling_rate"],
        random_sampler=False,
        transform=val_transform,
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # sample = next(iter(train_ds))
    # sample_video = sample[0]
    #
    # inv_transform = Compose([InverseNormalize(mean, std),
    #                          Lambda(lambda x: torch.clip((x * 255.0).to(torch.uint8), min=0, max=255)),
    #                          Lambda(lambda x: x.permute(1, 2, 3, 0))])
    #
    # sample_video = inv_transform(sample_video).numpy()
    # clip = ImageSequenceClip(list(sample_video), fps=30)
    # clip.write_videofile("check.mp4")

    x3d = LitX3DTransfer(model_name, num_classes)

    trainer = pl.Trainer(limit_train_batches=2, max_epochs=1)
    trainer.fit(model=x3d, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
