import torch

from surgidoc.vision.classes import CLASS_NAMES, IGNORE_INDEX
from surgidoc.vision.metrics import MIN_FRAMES_TO_SCORE, ConfusionMatrix

LIVER = CLASS_NAMES.index("liver")
BG = CLASS_NAMES.index("background")


def _frames(n, target_value, pred_value):
    return torch.full((n, 4, 4), pred_value), torch.full((n, 4, 4), target_value)


def test_perfect_prediction_scores_one():
    cm = ConfusionMatrix()
    pred, target = _frames(MIN_FRAMES_TO_SCORE, LIVER, LIVER)
    cm.update(pred, target)
    s = cm.scores()
    assert s["per_class"]["liver"]["dice"] == 1.0
    assert s["scored_classes"] == ["liver"]


def test_half_overlap():
    cm = ConfusionMatrix()
    target = torch.full((MIN_FRAMES_TO_SCORE, 4, 4), LIVER)
    target[:, :, :2] = BG
    pred = torch.full_like(target, LIVER)  # predicts liver everywhere
    cm.update(pred, target)
    liver = cm.scores()["per_class"]["liver"]
    assert abs(liver["iou"] - 0.5) < 1e-9
    assert abs(liver["dice"] - 2 / 3) < 1e-9


def test_rare_or_untrained_classes_are_not_scored():
    cm = ConfusionMatrix()
    cm.update(*_frames(MIN_FRAMES_TO_SCORE - 1, LIVER, LIVER))
    assert cm.scores()["per_class"]["liver"]["dice"] is None

    cm.update(*_frames(1, LIVER, LIVER))
    assert cm.scores(train_classes={"background"})["per_class"]["liver"]["dice"] is None
    assert cm.scores(train_classes={"liver"})["per_class"]["liver"]["dice"] == 1.0


def test_ignore_pixels_are_skipped():
    cm = ConfusionMatrix()
    pred = torch.full((1, 4, 4), BG)
    target = torch.full((1, 4, 4), IGNORE_INDEX)
    cm.update(pred, target)
    assert cm.matrix.sum() == 0
