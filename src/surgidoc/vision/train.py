"""Fine-tune a U-Net on CholecSeg8k and evaluate it on the held-out test videos.

    python -m surgidoc.vision.train --root /kaggle/input/cholecseg8k --out runs/unet_r34

Writes to --out: best.pt (best validation mean IoU), history.json (per epoch) and
results.json (per-class test scores of best.pt).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .classes import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from .dataset import CholecSeg8k
from .metrics import ConfusionMatrix
from .model import build_unet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="CholecSeg8k folder with videoXX dirs")
    p.add_argument("--split-file", type=Path, default=Path("configs/splits/cholecseg8k.json"))
    p.add_argument("--class-stats", type=Path, default=Path("reports/class_stats.json"))
    p.add_argument("--out", type=Path, default=Path("runs/unet"))
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="imagenet", help="'none' for random init")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dice-weight", type=float, default=0.7, help="loss = w*dice + (1-w)*ce")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--limit-batches", type=int, default=0, help="smoke test: batches per loop")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_classes(class_stats: Path, split_file: Path) -> set[str]:
    """Classes that occur in at least one training frame."""
    stats = json.loads(class_stats.read_text())
    videos = json.loads(split_file.read_text())["train"]
    return {c for c in CLASS_NAMES if any(stats[v]["frames_with_class"][c] > 0 for v in videos)}


class DiceCELoss(nn.Module):
    """Weighted soft Dice + cross-entropy. Dice keeps small classes from being drowned
    out by large ones; CE gives stable per-pixel gradients early in training."""

    def __init__(self, dice_weight: float) -> None:
        super().__init__()
        from segmentation_models_pytorch.losses import DiceLoss

        self.w = dice_weight
        self.dice = DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w * self.dice(logits, target) + (1 - self.w) * self.ce(logits, target)


@torch.no_grad()
def evaluate(model, loader, device, limit: int = 0) -> ConfusionMatrix:
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES)
    for i, (images, masks) in enumerate(loader):
        if limit and i >= limit:
            break
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        cm.update(logits.argmax(1), masks)
    return cm


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = (args.height, args.width)

    def loader(split: str, shuffle: bool) -> DataLoader:
        ds = CholecSeg8k(args.root, args.split_file, split, image_size=size)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle,
                          num_workers=args.workers, pin_memory=device.type == "cuda",
                          persistent_workers=args.workers > 0)

    train_loader, val_loader = loader("train", True), loader("val", False)
    seen = train_classes(args.class_stats, args.split_file)
    print(f"device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = build_unet(args.encoder, weights).to(device)
    criterion = DiceCELoss(args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.limit_batches or len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="surgidoc-seg", config=config)

    history, best_miou = [], float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for i, (images, masks) in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), masks)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        val = evaluate(model, val_loader, device, args.limit_batches).scores(seen)
        row = {"epoch": epoch, "train_loss": total_loss / steps_per_epoch,
               "val_mean_iou": val["mean_iou"], "val_mean_dice": val["mean_dice"],
               "lr": scheduler.get_last_lr()[0], "seconds": round(time.time() - start, 1)}
        history.append(row)
        print(json.dumps(row))
        if run:
            run.log(row)

        # No scorable class (only in tiny smoke runs) counts as -1, so best.pt always exists.
        miou = val["mean_iou"] if val["mean_iou"] is not None else -1.0
        if miou > best_miou:
            best_miou = miou
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                        "val": val, "class_names": CLASS_NAMES}, args.out / "best.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2))

    # Test once, with the checkpoint chosen on validation, never on test.
    ckpt = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, loader("test", False), device, args.limit_batches).scores(seen)
    results = {"best_epoch": ckpt["epoch"], "val": ckpt["val"], "test": test, "config": config}
    (args.out / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nbest epoch {ckpt['epoch']}  test mIoU {test['mean_iou']}  mDice {test['mean_dice']}")
    for name, s in test["per_class"].items():
        score = f"dice {s['dice']:.3f}  iou {s['iou']:.3f}" if s["dice"] is not None else "not scored"
        print(f"  {name:24s} frames {s['frames']:5d}  {score}")
    if run:
        run.summary.update({"test_mean_iou": test["mean_iou"], "test_mean_dice": test["mean_dice"]})
        run.finish()


if __name__ == "__main__":
    main()
