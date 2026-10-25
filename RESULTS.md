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

## Agent: grounding of report claims

Measured 2026-10-24 on the 3 test videos (`reports/eval_grounding.json`). Each run's claims
are checked twice with the same verifier: once against the vision models' output (what
the agent sees), once against facts rebuilt from CholecSeg8k's hand-drawn masks.

| Condition | Claims | Gradable | Grounded | Contradicted | Contradicted, measured class |
|---|---:|---:|---:|---:|---:|
| First draft, no verifier | 50 | 21 | 20 | 1 | 0 |
| After verifier and retries | 50 | 21 | 20 | 1 | 0 |

- **Gradable** claims say a structure is visible: structure claims and event claims that
  name a structure. Phase claims have no ground truth here; guideline and limitation claims
  make no observation. Claims are graded on every structure their text names.
- **The one contradiction** (video55): "connective tissue is detected in key frames". The
  U-Net saw it in 6 frames, the masks in none. Connective tissue has no measured test score,
  so the report already labels the claim unverified.
- **Verifier value: no measurable effect in these runs.** The verifier rejected 5 first-draft
  claims in video55, and Claude fixed them on retry. All 5 were true by the masks; they named
  instruments without listing them. In these 3 runs Claude wrote no false claim that the
  verifier could catch, so the two conditions are identical. Measuring what the verifier
  catches needs drafts with known errors (injected false claims), which this eval does not do.
- **Small sample.** 3 videos, 1 run each, 21 gradable claims. Not a precision estimate.

```bash
.venv/bin/python -m surgidoc.agent.graph run --source /path/to/cholecseg8k/video17 --thread eval-video17
.venv/bin/python -m surgidoc.evals.grounding --threads eval-video17 eval-video37 eval-video55
```

## Verifier: injected errors

Measured 2026-10-25 (`reports/eval_injected.json`, seed 0). For each of the 3 test runs,
code builds claims from that run's own facts, timeline and retrieved passages, each with a
known answer, and passes them to the verifier. Up to 5 cases per category and run.

| Category | What is wrong | n | Rejected |
|---|---|---:|---:|
| absent_structure | structure not visible in any cited frame | 15 | 15 |
| partial_support | structure visible in half the cited frames | 7 | 7 |
| wrong_phase | phase differs from the model's in every cited frame | 15 | 15 |
| unknown_frames | frame ids that do not exist | 15 | 15 |
| unknown_event | event id that does not exist | 15 | 15 |
| unlisted_structure | text names an absent structure the claim does not list | 15 | 15 |
| unretrieved_passage | cites a corpus passage not retrieved in the run | 15 | 15 |
| altered_quote | one word of a real quote swapped | 15 | 15 |
| quote_from_other_passage | quote is real but from a different passage than cited | 15 | 15 |
| guideline_with_observation | guideline claim cites frames | 15 | 15 |
| **All errors** | | **142** | **142** |
| *Blind spot:* vision_error | model sees the structure, the masks do not | 8 | 0 |
| *Blind spot:* misstated_guideline | verbatim quote, claim says the opposite | 15 | 0 |
| *Control:* true structure, phase and guideline claims | nothing | 45 | 0 |

Every rejection was for the intended reason (checked per category in the report file).

- **What 100% means.** The errors are built from templates that match the verifier's rules,
  so this shows the rules are implemented and hold on real run data, with no false alarms
  on true claims. It is not a catch rate for real model mistakes, which are messier.
- **What the verifier cannot catch.** Both blind spots pass every time, by design. The
  verifier trusts the vision output, so a segmentation error becomes a supported claim
  (7 of the 8 cases are on classes without a measured test score, which the report labels
  unverified; 1 is gallbladder). And it checks that a quote exists, not that the claim
  means what the quote says. Those two are left to the reviewer, which is why the report
  shows the quote next to each guideline claim.

```bash
.venv/bin/python -m surgidoc.evals.injected --threads eval-video17 eval-video37 eval-video55
```

## Judge: does each claim say only what its evidence says?

Measured 2026-10-25. The verifier checks references with code, not meaning. The judge
node (`src/surgidoc/agent/judge.py`, Claude Haiku 5.5, one call per draft) reads each claim
the verifier passed next to its evidence (model output for its frames and events, or its
quote and passage) and answers faithful, overstates or contradicts. Anything but faithful
goes back to the drafter.

**Planted claims** (`surgidoc.evals.injected --judge`, same cases as above):

| Category | n | Rejected by verifier | Rejected by verifier + judge |
|---|---:|---:|---:|
| Errors the verifier catches (10 kinds) | 142 | 142 | 142 |
| Misstated guideline (verbatim quote, opposite claim) | 15 | 0 | 15 |
| Overstated observation (an action or condition the models cannot see) | 15 | 0 | 15 |
| Vision error (model sees it, masks do not) | 8 | 0 | 0 |
| True claims (controls) | 45 | 0 | 0 |

**Real claims** (`surgidoc.evals.judge_real`, `configs/evals/judge_labels.json`). Claims
Claude wrote in agent runs, each labelled by me as faithful or overstating before scoring.

| Set | Claims | Overstating (my labels) | Caught | Missed | False alarms |
|---|---:|---:|---:|---:|---:|
| Tuning: test-video runs | 50 | 3 | 3 | 0 | 0 |
| Held out: validation-video runs | 33 | 3 | 1 | 2 | 0 |

- **What it catches:** actions, locations and spatial relations added to "in view"
  ("the L-hook is in use on the cystic region", "near the gallbladder"), and guidance
  turned around or stretched ("recommends videotaping the procedure", where the source asks
  for an existing videotape to be added to the report).
- **What it misses:** subtle distortions of a quote. "A complete critical view of safety is
  achieved in only about 50% of cases" drops one word from "easily obtained in only 50% of
  cases", which changes the meaning. And claims about the system itself it has no
  evidence for (video18 c13, borderline).
- **No false alarms** on 77 faithful real claims or 45 planted true claims.
- **Tuning history.** The first version flagged 7 of the tuning set's 50 claims, 3 of them
  wrongly, because the judge lacked frame ids, areas and the timeline. After adding them,
  11 flags, with reasons that often argued for "faithful": the schema put the verdict before
  the reason. Reason first, plus the passage as context for guideline claims, gave the
  numbers above. Hence the held-out set, drafted and labelled after these changes.
- **Caveats.** One labeller (me), 6 overstating claims in total, and an LLM judging an
  LLM of the same family. The drafter's prompt now also tells it not to state actions,
  so overstatements are rarer in the held-out drafts.

```bash
.venv/bin/python -m surgidoc.evals.injected --threads eval-video17 eval-video37 eval-video55 --judge
.venv/bin/python -m surgidoc.evals.judge_real --set heldout
```

## Agent: tool use in key-frame selection

Measured 2026-10-25 (`reports/eval_tool_use.json`). Key-frame selection is the one node
where Claude calls tools (`frame_stats`, `view_frames`) and decides what to inspect, so
there is no single correct call sequence. The eval reran selection 3 times on each test
run's facts (9 selections, Claude Haiku 5.5) and compared it with the node's deterministic
fallback rule (middle of each phase segment, start of each event) and with random frames.

| Selection | Frames | Segments covered | Event starts covered | Near-duplicate pairs | Validated structures per frame (masks) |
|---|---:|---:|---:|---:|---:|
| Agent (mean of 9) | 5.9 | 0.96 | 0.85 | 0.00 | 5.98 |
| Fallback rule | 7.0 | 1.00 | 1.00 | 0.00 | 5.98 |
| Random frames, same count | 5.9 | 0.90 | 0.24 | 0.79 | 5.84 |

Tool use, per selection: 1.9 tool calls over 2.2 turns, 4.8 of 12 allowed images viewed,
no call with an invalid frame id, no frame viewed twice, 98% of chosen frames inspected
before being chosen, fallback never needed. About 7,200 input tokens per selection.

- **Tool calls are clean.** Valid arguments, within budget, no repeats, and the agent looks
  at what it picks.
- **No measured benefit over the rule.** On every criterion that can be checked, the
  fallback rule does as well or better, with no API call. Most agent misses are 1-3 frame
  grasper flickers in video17, which a reader would likely skip too; two are real (the last
  phase segment of video17 in one repeat, a 12-frame L-hook event in video55 in another).
- **The value metric is weak.** Structures per frame barely separates any selection from
  random frames (5.98 vs 5.84): in CholecSeg8k most frames show the same few structures.
  Whether the agent's frames are better documentation would need a human rating.

```bash
.venv/bin/python -m surgidoc.evals.tool_use --threads eval-video17 eval-video37 eval-video55 --repeats 3
```

## Report: comparison with what a real operative report contains

Compiled 2026-10-25 (`configs/evals/report_checklist.json`, `reports/eval_checklist.json`).
No public dataset pairs Cholec80 or CholecSeg8k videos with surgeons' reports, so the
reports from the 3 test runs are compared with what such a report must contain. Four
sources, merged into 31 items:

- **DE:** a German operative report template for laparoscopic cholecystectomy (MIC-Manual
  der Endoskopischen Chirurgie, Kliniken Essen-Mitte, 2001; fictitious patient). Cited, not
  copied.
- **RCSI:** the Royal College of Surgeons of Ireland's 14 criteria for operative notes.
- **WSES:** the WSES 2020 guideline's minimum content of the operative report.
- **DSS:** the Dutch Surgical Society's 6 key steps.

| Source | Items | Covered | Partial | Visible in a full recording, not reported | Needs the record, not the video |
|---|---:|---:|---:|---:|---:|
| German template | 22 | 1 | 4 | 9 | 8 |
| RCSI | 14 | 0 | 3 | 1 | 10 |
| WSES | 11 | 1 | 2 | 5 | 3 |
| DSS | 6 | 0 | 1 | 4 | 1 |
| **All, deduplicated** | **31** | **1** | **5** | **11** | **14** |

- **Covered:** the energy device (L-hook electrocautery in view, with times), in the 2 runs
  whose footage shows it.
- **Partial:** findings (which structures are visible, not their condition), Calot
  dissection (as a phase estimate, 2 of 3 runs), complications (possible bleeding only,
  unvalidated), key images (chosen for coverage, not for the steps a surgeon photographs),
  and sign-off (reviewer approval, no named signature).
- **Visible but not reported:** access, adhesiolysis, the critical view of safety, clipping,
  gallbladder dissection, retrieval, haemostasis, anatomical variations, conversion, drain,
  operative time. Some need models SurgiDoc does not have (CVS, actions, clip counting);
  most are simply not in the 10-25 s CholecSeg8k fragments. The critical view of safety is
  handed to the surgeon as a limitation in all 3 reports.
- **Needs the record:** patient, date, team, anaesthesia, urgency, diagnosis and codes,
  blood loss volume, closure, histology, implants, cholangiography, post-operative orders.

**How the reports differ, beyond coverage:**

- **The German report narrates actions and states judgements.** It is one paragraph of what
  the surgeon did, in order ("Einbringen der Trokare", "Darstellung des Ductus cysticus"),
  with verdicts such as "können zweifelsfrei identifiziert werden" (can be identified
  beyond doubt). That verdict is the central safety statement of the report, and it is the
  one SurgiDoc defers to the surgeon.
- **SurgiDoc lists observations, with their uncertainty.** What is visible when, with frame
  times, phase estimates labelled as estimates, unvalidated classes labelled unverified,
  guideline passages quoted, and an explicit list of what it could not determine. The
  template has none of these.
- **So the output is a draft of the video part of the report.** It can supply timestamps,
  instrument use and key frames that a surgeon would otherwise reconstruct from memory, plus
  reminders of what the guidelines ask to be documented. It does not replace the report.
- **One wording problem found while comparing.** video55's report says the L-hook "is in
  use on the cystic region ... while the grasper holds the tissue". The vision models only
  show both instruments in view; "in use", "cystic region" and "holds" are Claude's
  interpretation. The verifier checks which structures a claim names, not its verbs, so
  this passed.

**Caveats.** The categories are my judgement from the system's design, not a surgeon's.
The German template is a single example from 2001. The covered and partial items were
checked against the 3 test runs' actual claims; the rest were not, by definition.

```bash
.venv/bin/python -m surgidoc.evals.checklist --threads eval-video17 eval-video37 eval-video55
```

## Guideline retrieval

Measured 2026-10-24 (`reports/eval_retrieval.json`). 27 hand-written questions over the
219-passage corpus, each labelled with the passages that answer it
(`configs/evals/retrieval_questions.json`). Every ranking reorders the same 25 dense
candidates from `bge-small-en-v1.5`; the agent keeps the top 3 per query.

| Ranking | MRR | Hit@1 | Hit@3 | Recall@3 | Hit@5 | Recall@5 | CPU s / query |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0.66 | 0.56 | 0.70 | 0.53 | 0.78 | 0.58 | <0.1 |
| + bge-reranker-base | 0.60 | 0.48 | 0.59 | 0.38 | 0.85 | 0.58 | 6.6 |
| + ms-marco-MiniLM-L-6-v2 | 0.73 | 0.59 | 0.85 | 0.59 | 0.93 | 0.67 | 1.1 |
| + bge-reranker-v2-m3 | 0.83 | 0.74 | 0.89 | 0.67 | 0.93 | 0.69 | 24.9 |

- **The reranker the agent was built with makes retrieval worse.** bge-reranker-base gives
  near-1.0 scores to most passages about cholecystectomy complications, so it reorders
  passages that are related but do not answer the question above the ones that do.
- **The agent now uses bge-reranker-v2-m3** (`surgidoc.retrieval.index.RERANK_MODEL`). The
  grounding runs above used bge-reranker-base; the reranker only affects guideline claims,
  which the grounding eval does not grade.
- **Dense recall is the ceiling.** For q02 (what to do when the critical view of safety
  cannot be achieved) none of the 4 answering passages is in the dense top 25.
- **Caveats.** I wrote and labelled the questions myself, while reading the corpus,
  which favours wording overlap. Five labels were added after a first run, for passages a
  ranker returned that do answer the question on a full read. The rerankers were compared
  on the same 27 questions, with no held-out set.

```bash
.venv/bin/python -m surgidoc.evals.retrieval --rerankers BAAI/bge-reranker-base \
    cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-v2-m3
```
