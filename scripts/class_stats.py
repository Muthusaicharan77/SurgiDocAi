"""Per-video class statistics for CholecSeg8k.

For every video: frame count, pixel count per class, and the number of frames in
which each class appears. Used to choose a train/val/test split in which every
class (especially the rare cystic duct and hepatic vein) can actually be scored.

    python scripts/class_stats.py --root /path/to/cholecseg8k --out reports/class_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from surgidoc.vision.classes import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, encode_mask
from surgidoc.vision.dataset import list_frames


def frame_counts(args: tuple[str, str]) -> tuple[str, np.ndarray]:
    video_id, mask_path = args
    labels = encode_mask(np.array(Image.open(mask_path)))
    # Bin IGNORE_INDEX into an extra last slot.
    labels = np.where(labels == IGNORE_INDEX, NUM_CLASSES, labels)
    return video_id, np.bincount(labels.ravel(), minlength=NUM_CLASSES + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/class_stats.json"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    frames = list_frames(args.root)
    pixels = defaultdict(lambda: np.zeros(NUM_CLASSES + 1, dtype=np.int64))
    present = defaultdict(lambda: np.zeros(NUM_CLASSES + 1, dtype=np.int64))
    n_frames = defaultdict(int)

    jobs = [(f.video_id, str(f.mask_path)) for f in frames]
    with Pool(args.workers) as pool:
        for video_id, counts in pool.imap_unordered(frame_counts, jobs, chunksize=32):
            pixels[video_id] += counts
            present[video_id] += counts > 0
            n_frames[video_id] += 1

    names = CLASS_NAMES + ["ignore"]
    report = {
        video: {
            "frames": n_frames[video],
            "pixels": dict(zip(names, pixels[video].tolist())),
            "frames_with_class": dict(zip(names, present[video].tolist())),
        }
        for video in sorted(n_frames)
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    total = sum(pixels.values())
    print(f"{len(frames)} frames, {len(report)} videos -> {args.out}")
    for name, share in sorted(zip(names, total / total.sum()), key=lambda x: -x[1]):
        print(f"  {name:24s} {share:7.2%}")


if __name__ == "__main__":
    main()
