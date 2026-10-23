import json

import numpy as np
import pytest
from PIL import Image

from surgidoc.vision.classes import CLASS_NAMES, IGNORE_INDEX, encode_mask
from surgidoc.vision.dataset import IMAGE_SUFFIX, MASK_SUFFIX, CholecSeg8k


def test_encode_mask_known_values():
    grey = np.array([[50, 21, 22], [25, 33, 5]], dtype=np.uint8)
    names = [CLASS_NAMES[i] for i in encode_mask(grey).ravel()]
    assert names == ["background", "liver", "gallbladder", "cystic_duct", "hepatic_vein",
                     "liver_ligament"]


def test_encode_mask_unknown_and_unlabeled_are_ignored():
    grey = np.array([[255, 0, 99]], dtype=np.uint8)
    assert (encode_mask(grey) == IGNORE_INDEX).all()


def test_encode_mask_accepts_three_channels():
    grey = np.full((2, 2, 3), 21, dtype=np.uint8)
    assert (encode_mask(grey) == CLASS_NAMES.index("liver")).all()


@pytest.fixture
def fake_root(tmp_path):
    """One video, one frame: left half red/liver, right half black/background."""
    clip = tmp_path / "video99" / "video99_00000"
    clip.mkdir(parents=True)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :32, 0] = 255
    mask = np.full((48, 64, 3), 50, dtype=np.uint8)
    mask[:, :32] = 21
    Image.fromarray(image).save(clip / f"frame_0{IMAGE_SUFFIX}")
    Image.fromarray(mask).save(clip / f"frame_0{MASK_SUFFIX}")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["video99"], "test": []}))
    return tmp_path, split


def test_flip_keeps_image_and_mask_aligned(fake_root):
    root, split = fake_root
    ds = CholecSeg8k(root, split, "train", image_size=(48, 64), augment=True)
    liver = CLASS_NAMES.index("liver")
    for _ in range(20):  # enough draws to hit both flip outcomes
        image, mask = ds[0]
        red_side = image[0] > 0  # normalised red channel is positive only where red
        assert ((mask == liver) == red_side).float().mean() > 0.95
