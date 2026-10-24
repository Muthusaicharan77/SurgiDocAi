"""Evaluate a trained segmentation checkpoint on one split.

    python -m surgidoc.vision.evaluate --root /path/to/cholecseg8k \
        --checkpoint weights/seg/unet_r34_best.pt --split test --out reports/seg_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import CholecSeg8k
from .model import load_checkpoint
from .train import evaluate, train_classes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--split-file", type=Path, default=Path("configs/splits/cholecseg8k.json"))
    p.add_argument("--class-stats", type=Path, default=Path("reports/class_stats.json"))
    p.add_argument("--out", type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, str(device))
    ds = CholecSeg8k(args.root, args.split_file, args.split,
                     image_size=(config["height"], config["width"]), augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    scores = evaluate(model, loader, device).scores(train_classes(args.class_stats, args.split_file))

    result = {"checkpoint": str(args.checkpoint), "split": args.split, "frames": len(ds), **scores}
    print(f"{args.split}: {len(ds)} frames  mIoU {scores['mean_iou']:.4f}  "
          f"mDice {scores['mean_dice']:.4f}")
    for name, s in scores["per_class"].items():
        score = f"dice {s['dice']:.3f}  iou {s['iou']:.3f}" if s["dice"] is not None else "not scored"
        print(f"  {name:24s} frames {s['frames']:5d}  {score}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
