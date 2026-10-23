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
- [ ] Train/val/test split by video (`configs/splits/`)
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
