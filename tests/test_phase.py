import numpy as np
import torch
from PIL import Image

from surgidoc.vision.phase import (
    PHASES,
    PhaseClassifier,
    _FrameClassifier,
    crop_black_margin,
    preprocess,
    smooth,
    to_labels,
)


def test_checkpoint_layout_loads_strictly(tmp_path):
    # A state_dict with Trans-SVNet's key names must load with no missing/extra keys.
    path = tmp_path / "w.pth"
    torch.save(_FrameClassifier().state_dict(), path)
    clf = PhaseClassifier(path)
    probs = clf.predict_proba([Image.new("RGB", (854, 480), (120, 60, 50))] * 3)
    assert probs.shape == (3, len(PHASES))
    assert np.allclose(probs.sum(1), 1, atol=1e-5)


def test_crop_black_margin_removes_border():
    image = np.zeros((480, 854, 3), dtype=np.uint8)
    image[40:440, 100:754] = 150  # bright endoscope view inside a black border
    cropped = crop_black_margin(Image.fromarray(image))
    w, h = cropped.size
    assert abs(w - 654) < 12 and abs(h - 400) < 12


def test_preprocess_shape():
    assert preprocess(Image.new("RGB", (854, 480), (100, 50, 40))).shape == (3, 224, 224)


def test_smooth_removes_single_frame_flicker():
    probs = np.tile(np.eye(7)[0], (9, 1))
    probs[4] = np.eye(7)[3]  # one stray frame
    labels = [name for name, _ in to_labels(smooth(probs, window=5))]
    assert labels == ["preparation"] * 9


def test_smooth_keeps_rows_normalised_and_window_one_is_identity():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(7), size=20)
    assert np.allclose(smooth(probs, 7).sum(1), 1)
    assert np.array_equal(smooth(probs, 1), probs)
