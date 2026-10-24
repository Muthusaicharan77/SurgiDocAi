"""Segmentation metrics from a confusion matrix accumulated over a whole split.

Dice and IoU are computed per class over all pixels of the split (not averaged
per image), so a frame where a class is absent cannot produce a 0/0.
"""

from __future__ import annotations

import torch

from .classes import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES

# A class is scored on a split only if the split has at least this many frames
# containing it. Below that, one or two frames would decide the score.
MIN_FRAMES_TO_SCORE = 20


class ConfusionMatrix:
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        self.frames_with_class = torch.zeros(num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: (B, H, W) class indices. Pixels with IGNORE_INDEX are skipped."""
        pred, target = pred.cpu(), target.cpu()
        for t in target:
            present = torch.unique(t[t != IGNORE_INDEX])
            self.frames_with_class[present] += 1
        valid = target != IGNORE_INDEX
        idx = target[valid] * self.num_classes + pred[valid]
        self.matrix += torch.bincount(idx, minlength=self.num_classes**2).view(
            self.num_classes, self.num_classes
        )

    def scores(self, train_classes: set[str] | None = None) -> dict:
        """Per-class Dice/IoU plus means over the scorable classes.

        A class is scorable if it occurs in at least MIN_FRAMES_TO_SCORE frames of
        this split and, when train_classes is given, also in the training videos.
        """
        tp = self.matrix.diag().double()
        fp = self.matrix.sum(0).double() - tp
        fn = self.matrix.sum(1).double() - tp
        dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
        iou = tp / (tp + fp + fn).clamp(min=1)

        per_class, scored = {}, []
        for i, name in enumerate(CLASS_NAMES[: self.num_classes]):
            ok = self.frames_with_class[i] >= MIN_FRAMES_TO_SCORE
            if train_classes is not None:
                ok = ok and name in train_classes
            per_class[name] = {
                "frames": int(self.frames_with_class[i]),
                "dice": float(dice[i]) if ok else None,
                "iou": float(iou[i]) if ok else None,
            }
            if ok:
                scored.append(i)
        return {
            "per_class": per_class,
            "mean_dice": float(dice[scored].mean()) if scored else None,
            "mean_iou": float(iou[scored].mean()) if scored else None,
            "scored_classes": [CLASS_NAMES[i] for i in scored],
        }
