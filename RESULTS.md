# Results

## Segmentation: U-Net on CholecSeg8k

Measured 2026-10-24. Numbers come from `reports/seg_test.json`, which anyone can regenerate
from the checkpoint with the command at the bottom.

### Setup

| | |
|---|---|
| Data | CholecSeg8k: 8,080 frames from 17 Cholec80 videos, 13 classes |
| Split | By whole video (`configs/splits/cholecseg8k.json`): train 12 videos / 6,640 frames, val 2 / 400, test 3 / 1,040 |
| Test videos | video17, video37, video55 (never seen in training or model selection) |
| Model | U-Net, ResNet34 encoder with ImageNet weights (`segmentation_models_pytorch`) |
| Input | 256 x 448 (roughly the original 480 x 854 aspect ratio) |
| Loss | 0.7 x soft Dice + 0.3 x cross-entropy, unlabeled pixels ignored |
| Training | 30 epochs, AdamW, one-cycle LR (max 3e-4), batch 16, mixed precision, seed 42, Kaggle T4 |
| Selection | Best validation mean IoU: epoch 14 (val mIoU 0.781, mDice 0.842) |

### Test results (3 unseen videos, 1,040 frames)

| Class | Test frames containing it | Dice | IoU |
|---|---:|---:|---:|
| Background (black border) | 1,040 | 0.996 | 0.992 |
| Abdominal wall | 1,040 | 0.908 | 0.831 |
| Fat | 1,040 | 0.863 | 0.759 |
| Gallbladder | 1,039 | 0.812 | 0.684 |
| Liver | 1,040 | 0.811 | 0.682 |
| Grasper | 960 | 0.767 | 0.622 |
| L-hook electrocautery | 326 | 0.720 | 0.562 |
| Gastrointestinal tract | 535 | 0.669 | 0.503 |
| **Mean over these 8 classes** | | **0.818** | **0.704** |

Dice and IoU are computed per class from one confusion matrix over all test pixels
(`src/surgidoc/vision/metrics.py`), not averaged per image.

### Not scored

A class is scored only if it occurs in the training videos and in at least 20 test frames.
With a split by whole video, five classes fail that:

| Class | Why |
|---|---|
| Cystic duct | 2 test frames. Nearly all cystic duct footage in CholecSeg8k is one ~10 s scene in video01 |
| Blood | 0 test frames (only video01 has it) |
| Hepatic vein | 0 test frames (only video12) |
| Liver ligament | 0 test frames (only video09) |
| Connective tissue | 0 test frames |

The model is trained on all 13 classes, but there is no evidence here about how well it
segments these five. Downstream components must treat them as unverified.

### Caveats

- **Small test set.** Three videos, and CholecSeg8k covers only short bursts of each
  (10-25 s, mostly from the first minutes of surgery). Consecutive frames are nearly
  identical, so 1,040 frames carry far less information than 1,040 independent images.
- **One run, one seed.** No variance estimate.
- **The mean flatters.** It covers the 8 classes that could be scored, which are the large or
  common ones. The thin, rare structures that matter most clinically are the ones not scored.
- **Background is easy.** It is the black border outside the endoscope view. Without it, the
  mean over the remaining 7 classes is Dice 0.793 / IoU 0.663.

### Reproduce

```bash
.venv/bin/python -m surgidoc.vision.train --root /path/to/cholecseg8k --out runs/unet_r34   # GPU
.venv/bin/python -m surgidoc.vision.evaluate --root /path/to/cholecseg8k \
    --checkpoint runs/unet_r34/best.pt --split test --out reports/seg_test.json
```

Checkpoint used here: `unet_r34_best.pt`, sha256
`9e0b3f4db8874ae9c4b3f94ec77c16e510c726d5ab910c338b3a39f6bf5d01a3` (not in git).

## Phase recognition

No accuracy is reported. See the README: the pretrained Trans-SVNet frame classifier is
used as published, 13 of the 17 CholecSeg8k videos are in its training set, and this project
uses no Cholec80 phase annotations.
