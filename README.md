# SurgiDoc

[![CI](https://github.com/Muthusaicharan77/SurgiDocAi/actions/workflows/ci.yml/badge.svg)](https://github.com/Muthusaicharan77/SurgiDocAi/actions/workflows/ci.yml)

Agentic documentation of laparoscopic cholecystectomy videos. The plan: segment the
anatomy and instruments in each frame, recognise the surgical phase, let an agent pick
the key frames, and draft a report in which every claim is checked against the vision
output before a surgeon reviews it.

> **Research demo on public data. Not a medical device, not for clinical use.**

This project extends a team project from the BlackForest Hackathon 2025 (SmartXHealth
track): [BFH-SxH](https://github.com/ChandrikaSethumadhavan/BFH-SxH).

## Status

Work in progress. Measured numbers are in [RESULTS.md](RESULTS.md); nothing is reported before it
is measured.

- [x] Corrected 13-class CholecSeg8k mapping (`src/surgidoc/vision/classes.py`)
- [x] Dataset with image/mask-consistent augmentation (`src/surgidoc/vision/dataset.py`)
- [x] Per-video class statistics (`reports/class_stats.json`)
- [x] Train/val/test split by whole video (`configs/splits/cholecseg8k.json`)
- [x] Training and evaluation script (`src/surgidoc/vision/train.py`), Kaggle notebook (`notebooks/train_kaggle.ipynb`)
- [x] U-Net fine-tuned on Kaggle GPU: test mIoU 0.704 / mDice 0.818 on 3 unseen videos, 8 scorable classes ([RESULTS.md](RESULTS.md))
- [x] Pretrained Trans-SVNet ResNet50 phase classifier with temporal smoothing (`src/surgidoc/vision/phase.py`)
- [x] Code-only agent nodes: frame sampling, perception, timeline, claim schema and deterministic verifier (`src/surgidoc/agent/`)
- [x] Guideline retrieval over a CC BY corpus (`src/surgidoc/retrieval/`)
- [x] Claude nodes (key-frame selection with tools, report drafting) and LangGraph wiring with verify/retry loop, human review interrupt and SQLite checkpointing (`src/surgidoc/agent/graph.py`)
- [x] Quote rule: every guideline claim carries a verbatim quote from its cited passage, checked by code (`src/surgidoc/agent/verify.py`) and shown to the reviewer
- [x] Evals: claim grounding against ground-truth masks, verifier on/off, retrieval hit@k / recall@k / MRR with three rerankers (`src/surgidoc/evals/`, [RESULTS.md](RESULTS.md))
- [x] Injected-error eval for the verifier: 142/142 planted errors rejected, 0/45 true claims rejected, 0/23 blind-spot cases caught ([RESULTS.md](RESULTS.md))
- [x] Tool-use eval for key-frame selection: clean tool calls, no measured gain over the deterministic fallback rule ([RESULTS.md](RESULTS.md))
- [x] Judge node: Claude checks that each claim says only what its evidence says; catches 30/30 planted meaning errors and 1 of 3 real overstatements on held-out runs, no false alarms ([RESULTS.md](RESULTS.md))
- [x] Checklist comparison with a German operative report template and three standards: 1 of 31 items covered, 5 partly, 11 visible in a full recording but not reported, 14 need the patient record ([RESULTS.md](RESULTS.md))
- [x] Langfuse tracing: a span per graph node, a generation per Claude call, verifier scores per trace (`src/surgidoc/tracing.py`)
- [ ] Trace screenshot from a real Langfuse instance

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

## Phase recognition

The per-frame ResNet50 classifier from Trans-SVNet (Gao et al., MICCAI 2021,
[xjgaocs/Trans-SVNet](https://github.com/xjgaocs/Trans-SVNet)) is used as published, with
no retraining. It predicts the 7 Cholec80 phases and was trained by its authors on Cholec80
videos 1-40. The weights are not redistributed here:

```bash
.venv/bin/pip install -e ".[download]"
.venv/bin/python scripts/download_weights.py   # fetches from the authors' Google Drive, checks sha256
```

Limitations:
- Trans-SVNet's temporal stages (TeCNO + transformer) are not used. They read a whole
  surgery at 1 fps; CholecSeg8k has at most about 25 s of continuous footage per video.
  Per-frame probabilities are smoothed with a centred moving average instead.
- No phase accuracy is reported. 13 of the 17 CholecSeg8k videos are in the phase model's
  training set, and this project uses no Cholec80 phase annotations.

## Running the agent

```bash
.venv/bin/pip install -e ".[agent,download]"
echo "ANTHROPIC_API_KEY=..." > .env            # git-ignored
.venv/bin/python scripts/download_weights.py   # phase model; segmentation weights from your own training run
.venv/bin/python scripts/fetch_guidelines.py && .venv/bin/python -m surgidoc.retrieval.index build

.venv/bin/python -m surgidoc.agent.graph run --source /path/to/cholecseg8k/video37
.venv/bin/python -m surgidoc.agent.graph review --thread <id> --approve [--remove c3] [--notes "..."]
```

The run pauses before the report is written and prints the verified claims for review.
The model is Claude Haiku 5.5 by default (`SURGIDOC_MODEL` to change it).

## Tracing (optional)

Runs can be traced in [Langfuse](https://langfuse.com), Cloud or self-hosted. Tracing is off
unless keys are set:

```bash
.venv/bin/pip install -e ".[trace]"
cat >> .env <<'KEYS'
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
KEYS
```

Each `run` and `review` becomes one trace: a span per graph node, and a generation per
Claude call with model, token usage, input and output. Frame images are replaced by a
placeholder before anything is sent. The thread id is the Langfuse session, so a run and
its review appear together. Each run trace is scored with the verifier pass rate, the
number of drafts, the claims dropped and whether key-frame selection fell back to the rule.
