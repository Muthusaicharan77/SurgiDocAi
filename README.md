# SurgiDoc

Agentic documentation of laparoscopic cholecystectomy videos. The plan: segment the
anatomy and instruments in each frame, recognise the surgical phase, let an agent pick
the key frames, and draft a report in which every claim is checked against the vision
output before a surgeon reviews it.

> **Research demo on public data. Not a medical device, not for clinical use.**

This project extends a team project from the BlackForest Hackathon 2025 (SmartXHealth
track): [BFH-SxH](https://github.com/ChandrikaSethumadhavan/BFH-SxH).

## Status

Work in progress. Numbers go in `RESULTS.md` once they are measured; nothing is reported
before then.

- [x] Corrected 13-class CholecSeg8k mapping (`src/surgidoc/vision/classes.py`)
- [x] Dataset with image/mask-consistent augmentation (`src/surgidoc/vision/dataset.py`)
- [x] Per-video class statistics (`reports/class_stats.json`)
- [x] Train/val/test split by whole video (`configs/splits/cholecseg8k.json`)
- [x] Training and evaluation script (`src/surgidoc/vision/train.py`), Kaggle notebook (`notebooks/train_kaggle.ipynb`)
- [ ] U-Net fine-tuning on Kaggle GPU, per-class Dice and IoU
- [ ] Pretrained Trans-SVNet phase model
- [ ] LangGraph agent, verifier, evals

## Setup

```bash
python -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Data

[CholecSeg8k](https://www.kaggle.com/datasets/newslab/cholecseg8k) (Hong et al., 2020):
8,080 frames from 17 Cholec80 videos with 13-class masks. It is not redistributed here.
Download it and point the scripts at the extracted folder:

```bash
.venv/bin/python scripts/class_stats.py --root /path/to/cholecseg8k --out reports/class_stats.json
```

## Training

On a GPU (Kaggle notebook in `notebooks/train_kaggle.ipynb`):

```bash
python -m surgidoc.vision.train --root /path/to/cholecseg8k --out runs/unet_r34
```

Classes are scored on a split only if they occur in the training videos and in at least
20 frames of that split. With the committed split, cystic duct, blood, hepatic vein, liver
ligament and connective tissue are trained but not scored on test.
