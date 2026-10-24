"""U-Net segmentation model and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .classes import NUM_CLASSES


def build_unet(encoder: str = "resnet34", encoder_weights: str | None = "imagenet") -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model from a checkpoint written by train.py. Returns (model, config)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    config = ckpt["config"]
    model = build_unet(config["encoder"], encoder_weights=None)
    model.load_state_dict(ckpt["model"])  # strict: a mismatch should fail loudly
    return model.to(device).eval(), config
