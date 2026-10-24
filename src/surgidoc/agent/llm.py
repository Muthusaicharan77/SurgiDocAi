"""Shared helpers for the nodes that call Claude (Anthropic Python SDK)."""

from __future__ import annotations

import base64
import io
import os
from functools import cache
from pathlib import Path

from PIL import Image

# One place to change the model. Haiku 5.5: cheapest current model with vision,
# tool use and structured output.
MODEL = os.environ.get("SURGIDOC_MODEL", "claude-haiku-5-5")
EFFORT = os.environ.get("SURGIDOC_EFFORT", "medium")

# Frames are sent at half resolution (427x240): enough to see the scene, about a
# quarter of the image tokens of the full 854x480 frame.
IMAGE_WIDTH = 427


@cache
def client():
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(Path(".env"))
    return anthropic.Anthropic()


def image_block(path: str | Path) -> dict:
    image = Image.open(path).convert("RGB")
    image = image.resize((IMAGE_WIDTH, round(image.height * IMAGE_WIDTH / image.width)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def usage_row(node: str, response) -> dict:
    u = response.usage
    return {"node": node, "model": MODEL, "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens}
