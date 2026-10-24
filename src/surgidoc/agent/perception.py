"""Nodes sample_frames and perceive: frames in, per-frame facts out.

Input is a folder of frames laid out like CholecSeg8k (a video folder with clip
subfolders, or a single clip folder), files named frame_<n>_endo.png where n is
the frame index in the original 25 fps video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage

from surgidoc.vision.classes import CLASS_NAMES
from surgidoc.vision.dataset import IMAGE_SUFFIX, IMAGENET_MEAN, IMAGENET_STD
from surgidoc.vision.model import load_checkpoint
from surgidoc.vision.phase import PHASES, PhaseClassifier, smooth

from .facts import FrameFacts, StructureFacts

FPS = 25
_FRAME_RE = re.compile(r"frame_(\d+)" + re.escape(IMAGE_SUFFIX) + "$")


@dataclass(frozen=True)
class FrameRef:
    frame_number: int
    path: Path

    @property
    def time_s(self) -> float:
        return self.frame_number / FPS


def list_frames(source: Path) -> list[FrameRef]:
    """All frames under source, in video order."""
    refs = []
    for path in Path(source).rglob(f"*{IMAGE_SUFFIX}"):
        m = _FRAME_RE.search(path.name)
        if m:
            refs.append(FrameRef(int(m.group(1)), path))
    if not refs:
        raise ValueError(f"no frame_<n>{IMAGE_SUFFIX} files under {source}")
    return sorted(refs, key=lambda r: r.frame_number)


def sample_frames(source: Path, every_n: int = 5) -> list[FrameRef]:
    """Node sample_frames: every Nth frame (5 -> 5 fps from 25 fps footage)."""
    return list_frames(source)[::every_n]


def contiguous_runs(times: list[float], max_gap_s: float) -> list[range]:
    """Index ranges with no time gap larger than max_gap_s between neighbours.

    CholecSeg8k has bursts of footage minutes apart; smoothing or grouping must
    never reach across such a break.
    """
    runs, start = [], 0
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > max_gap_s:
            runs.append(range(start, i))
            start = i
    runs.append(range(start, len(times)))
    return runs


class Perceiver:
    """Node perceive: U-Net segmentation and phase classification for each frame."""

    def __init__(self, seg_checkpoint: Path, phase_weights: Path, device: str = "cpu",
                 smoothing_s: float = 2.0) -> None:
        self.device = device
        self.seg_model, cfg = load_checkpoint(seg_checkpoint, device)
        self.seg_size = [cfg["height"], cfg["width"]]
        self.phase = PhaseClassifier(phase_weights, device)
        self.smoothing_s = smoothing_s

    @torch.no_grad()
    def segment(self, images: list[Image.Image], batch_size: int = 16) -> list[np.ndarray]:
        masks = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([
                TF.normalize(TF.to_tensor(TF.resize(im, self.seg_size, antialias=True)),
                             IMAGENET_MEAN, IMAGENET_STD)
                for im in images[i : i + batch_size]
            ])
            masks.extend(self.seg_model(batch.to(self.device)).argmax(1).cpu().numpy())
        return masks

    def run(self, frames: list[FrameRef]) -> list[FrameFacts]:
        images = [Image.open(f.path).convert("RGB") for f in frames]
        masks = self.segment(images)
        probs = self.phase.predict_proba(images)

        times = [f.time_s for f in frames]
        step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        window = max(1, round(self.smoothing_s / step))
        for run in contiguous_runs(times, max_gap_s=3 * step):
            probs[run.start : run.stop] = smooth(probs[run.start : run.stop], window)

        facts = []
        for i, (ref, mask, p) in enumerate(zip(frames, masks, probs)):
            structures = {}
            for c in np.unique(mask):
                region = mask == c
                structures[CLASS_NAMES[c]] = StructureFacts(
                    area=round(float(region.mean()), 5),
                    components=int(ndimage.label(region)[1]),
                )
            facts.append(FrameFacts(
                frame_id=i, frame_number=ref.frame_number, time_s=ref.time_s,
                source=str(ref.path), phase=PHASES[int(p.argmax())],
                phase_confidence=round(float(p.max()), 4), structures=structures,
            ))
        return facts
