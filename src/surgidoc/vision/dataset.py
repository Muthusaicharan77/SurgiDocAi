"""CholecSeg8k dataset for semantic segmentation.

Layout on disk: <root>/videoXX/videoXX_NNNNN/frame_K_endo.png plus
frame_K_endo_watershed_mask.png next to it. Splits are by video, read from a
committed JSON file, so no frames from one video can leak across splits.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode

from .classes import encode_mask

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_SUFFIX = "_endo.png"
MASK_SUFFIX = "_endo_watershed_mask.png"


@dataclass(frozen=True)
class Frame:
    image_path: Path
    mask_path: Path
    video_id: str
    clip_id: str


def list_frames(root: Path, videos: list[str] | None = None) -> list[Frame]:
    """All (image, mask) pairs under root, optionally restricted to some videos."""
    frames = []
    for video_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if videos is not None and video_dir.name not in videos:
            continue
        for clip_dir in sorted(p for p in video_dir.iterdir() if p.is_dir()):
            for image_path in sorted(clip_dir.glob(f"*{IMAGE_SUFFIX}")):
                mask_path = image_path.with_name(image_path.name.replace(IMAGE_SUFFIX, MASK_SUFFIX))
                if mask_path.exists():
                    frames.append(Frame(image_path, mask_path, video_dir.name, clip_dir.name))
    return frames


def load_split(split_file: Path, split: str) -> list[str]:
    with open(split_file, encoding="utf-8") as fp:
        return json.load(fp)[split]


class CholecSeg8k(Dataset):
    def __init__(
        self,
        root: Path,
        split_file: Path,
        split: str,
        image_size: tuple[int, int] = (256, 448),  # (H, W); keeps the 480x854 aspect roughly
        augment: bool | None = None,
    ) -> None:
        self.frames = list_frames(root, load_split(split_file, split))
        if not self.frames:
            raise ValueError(f"no frames found for split '{split}' under {root}")
        self.image_size = list(image_size)
        self.augment = (split == "train") if augment is None else augment
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self.frames[index]
        image = Image.open(frame.image_path).convert("RGB")
        mask = Image.fromarray(encode_mask(np.array(Image.open(frame.mask_path))))

        # Geometric transforms go through one code path for image and mask, so
        # they can never disagree. Nearest-neighbour keeps mask values discrete.
        image = TF.resize(image, self.image_size, InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, self.image_size, InterpolationMode.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            image = self.jitter(image)  # photometric: image only

        image = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return image, torch.from_numpy(np.array(mask, dtype=np.int64))
