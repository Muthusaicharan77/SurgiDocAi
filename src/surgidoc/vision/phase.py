"""Surgical phase recognition with the pretrained Trans-SVNet ResNet50 frame classifier.

Weights: Gao et al., "Trans-SVNet: Accurate Phase Recognition from Surgical Videos via
Hybrid Embedding Aggregation Transformer", MICCAI 2021 (github.com/xjgaocs/Trans-SVNet),
trained on Cholec80 videos 1-40. Download with scripts/download_weights.py.

Only the per-frame stage is used. Trans-SVNet's temporal stages (TeCNO + transformer)
need a whole surgery at 1 fps, and CholecSeg8k has at most ~25 s of continuous
footage per video. Per-frame predictions are smoothed over a window instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch import nn

# Cholec80 phase order used by Trans-SVNet/TMRNet (phase_dict_key in get_paths_labels.py).
PHASES = [
    "preparation",
    "calot_triangle_dissection",
    "clipping_cutting",
    "gallbladder_dissection",
    "gallbladder_packaging",
    "cleaning_coagulation",
    "gallbladder_retraction",
]

# Cholec80 statistics used by Trans-SVNet instead of ImageNet's.
CHOLEC80_MEAN = (0.41757566, 0.26098573, 0.25888634)
CHOLEC80_STD = (0.21938758, 0.1983, 0.19342837)

_RESNET_PARTS = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4",
                 "avgpool"]


def crop_black_margin(image: Image.Image, threshold: int = 15) -> Image.Image:
    """Crop the black border around the endoscope view, like TMRNet's change_size().

    The mask is computed on a downscaled copy (the median filter is slow at full
    size) and the bounding box is scaled back.
    """
    small = image.convert("L")
    scale = 256 / small.width
    small = small.resize((256, max(1, round(small.height * scale))))
    mask = np.array(small.point(lambda v: 255 if v > threshold else 0).filter(
        ImageFilter.MedianFilter(5)))
    mask[:, :3] = 0  # TMRNet skips the outer 10 px columns at full size
    mask[:, -3:] = 0
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return image
    box = (cols.min() / scale, rows.min() / scale, (cols.max() + 1) / scale,
           (rows.max() + 1) / scale)
    return image.crop(tuple(round(v) for v in box))


def preprocess(image: Image.Image) -> torch.Tensor:
    """Trans-SVNet test transform (crop_type 1): resize 250x250, centre crop 224."""
    image = crop_black_margin(image.convert("RGB"))
    image = TF.center_crop(TF.resize(image, [250, 250], antialias=True), [224, 224])
    return TF.normalize(TF.to_tensor(image), CHOLEC80_MEAN, CHOLEC80_STD)


class _FrameClassifier(nn.Module):
    """ResNet50 trunk + 2048-512-7 head, laid out like Trans-SVNet's resnet_lstm."""

    def __init__(self) -> None:
        super().__init__()
        resnet = torchvision.models.resnet50(weights=None)
        self.share = nn.Sequential()
        for name in _RESNET_PARTS:
            self.share.add_module(name, getattr(resnet, name))
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, len(PHASES)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.share(x).flatten(1))


class PhaseClassifier:
    def __init__(self, weights: Path, device: str = "cpu") -> None:
        self.device = device
        self.model = _FrameClassifier()
        # weights_only=True: the file is a plain state_dict, so refuse anything that
        # would need arbitrary unpickling. Strict load: every key must match.
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(device).eval()

    @torch.no_grad()
    def predict_proba(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Per-frame phase probabilities, shape (N, 7)."""
        out = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([preprocess(im) for im in images[i : i + batch_size]])
            out.append(self.model(batch.to(self.device)).softmax(1).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, len(PHASES)), np.float32)


def smooth(probs: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average of per-frame probabilities over `window` frames.

    Single frames flicker between phases (motion blur, smoke, the camera briefly
    pointing elsewhere); a phase lasts minutes, so averaging removes most flicker.
    Centred, not causal: documentation runs after the surgery, so future frames are
    available. Near the ends the window shrinks rather than padding.
    """
    if window <= 1 or len(probs) == 0:
        return probs
    csum = np.cumsum(np.vstack([np.zeros((1, probs.shape[1])), probs]), axis=0)
    half = window // 2
    idx = np.arange(len(probs))
    lo, hi = np.maximum(idx - half, 0), np.minimum(idx + half + 1, len(probs))
    return (csum[hi] - csum[lo]) / (hi - lo)[:, None]


def to_labels(probs: np.ndarray) -> list[tuple[str, float]]:
    """(phase name, confidence) per frame."""
    return [(PHASES[i], float(p[i])) for p, i in zip(probs, probs.argmax(1))]
