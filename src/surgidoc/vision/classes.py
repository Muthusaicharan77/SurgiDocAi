"""CholecSeg8k class definitions.

The `*_endo_watershed_mask.png` files store one grey value per class (all three
channels are equal). The mapping below was checked against the colour masks.
Class order follows the CholecSeg8k paper (Hong et al., 2020), classes 0 to 12.
"""

from __future__ import annotations

import numpy as np

IGNORE_INDEX = 255  # unlabeled pixels; excluded from loss and metrics

# (class name, grey value in the watershed mask)
CLASSES: list[tuple[str, int]] = [
    ("background", 50),
    ("abdominal_wall", 11),
    ("liver", 21),
    ("gastrointestinal_tract", 13),
    ("fat", 12),
    ("grasper", 31),
    ("connective_tissue", 23),
    ("blood", 24),
    ("cystic_duct", 25),
    ("l_hook_electrocautery", 32),
    ("gallbladder", 22),
    ("hepatic_vein", 33),
    ("liver_ligament", 5),
]

CLASS_NAMES: list[str] = [name for name, _ in CLASSES]
NUM_CLASSES = len(CLASSES)

# 256-entry lookup table: grey value -> class index. Anything not listed,
# including 255, becomes IGNORE_INDEX.
_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _idx, (_, _value) in enumerate(CLASSES):
    _LUT[_value] = _idx


def encode_mask(mask: np.ndarray) -> np.ndarray:
    """Turn a watershed mask (H, W) or (H, W, 3) of grey values into class indices."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    return _LUT[mask]
