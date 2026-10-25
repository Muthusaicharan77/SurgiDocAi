"""Node judge: does each claim say only what its evidence says?

The verifier checks references and structure names with code, but not meaning: a
claim can quote a passage verbatim and still turn it around, or cite frames where
the hook is visible and add that it "is cutting the cystic plate". Here Claude
reads each claim that passed the verifier next to the evidence the verifier used
(the model output for its frames and events, or its quote) and judges whether the
claim stays within it. Claims judged to overstate or contradict their evidence are
sent back to the drafter with the reason, like any other verifier failure.

An LLM judging an LLM's output is itself unverified; its catch rate and false alarms
are measured on planted claims (surgidoc.evals.injected, RESULTS.md).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from surgidoc.tracing import generation
from surgidoc.vision.classes import CLASS_NAMES

from .claims import Claim, ClaimResult
from .facts import FrameFacts, Timeline, validated_classes
from .llm import EFFORT, MODEL, client, usage_row

ENABLED = os.environ.get("SURGIDOC_JUDGE", "1") != "0"

SYSTEM = """You check claims from a draft report on a laparoscopic cholecystectomy video. \
Each claim comes with the only evidence it may rest on. Research demo, not clinical use.

For each claim, answer:
- faithful: the claim says nothing the evidence does not show. Paraphrase, summary and \
cautious wording ("appears", "estimated") are fine.
- overstates: the claim adds something the evidence does not show, e.g. an action \
("cutting", "retracting", "dissecting"), a location ("on the cystic duct"), a condition \
("inflamed"), a cause, or certainty the evidence lacks. For guideline claims: the claim \
generalises, strengthens or extends what the quote says.
- contradicts: the claim says the opposite of the evidence.

Observation evidence is the output of vision models: which structures are visible in \
which frames, the phase estimate, and instrument events. "Visible" or "in view" is all \
the models can say; they do not see actions or what an instrument touches. \
Guideline evidence is the quoted passage. A limitation claim says what the system could \
not determine; it is faithful unless it asserts an observation.

Judge every claim id given. Write the reason first: compare the claim with the evidence \
and name any part that goes beyond it. Then give the verdict that follows from the reason: \
if the reason finds nothing beyond the evidence, the verdict is faithful. Close paraphrase \
and small differences of tone ("should" for "will") are faithful. Guideline claims may name \
the topic from the passage around the quote; what they say about it must be in the quote."""


class Verdict(BaseModel):
    # Reason before verdict, so the verdict follows from the reasoning (output order is
    # generation order).
    claim_id: str
    reason: str
    verdict: Literal["faithful", "overstates", "contradicts"]


class Judgement(BaseModel):
    verdicts: list[Verdict]


def evidence(c: Claim, facts: dict[int, FrameFacts], timeline: Timeline,
             passages: dict[str, str]) -> dict:
    """The evidence a claim may rest on, as the judge sees it."""
    if c.claim_type == "guideline":
        return {"quote": c.quote,
                "passage_for_context": " ".join(passages.get(p, "") for p in c.passage_ids)}
    if c.claim_type == "limitation":
        # What the system did determine, so a limitation can be checked against it.
        validated = validated_classes()
        detected = {n for f in facts.values() for n in f.structures if f.visible(n)}
        return {"classes_the_segmentation_model_has": [n for n in CLASS_NAMES
                                                       if n != "background"],
                "structures_detected_in_any_frame": sorted(detected - {"background"}),
                "classes_with_measured_accuracy": sorted(validated),
                "classes_without_measured_accuracy": sorted(
                    set(CLASS_NAMES) - validated - {"background"}),
                "phase_model": "no measured accuracy on this data",
                "phase_segments": [f"{g.phase} frames {g.start_frame}-{g.end_frame}"
                                   for g in timeline.segments],
                "footage_gaps": [f"between frames {a} and {b}" for a, b in timeline.gaps],
                "events": [f"{e.id} {e.kind} {e.detail} frames {e.start_frame}-{e.end_frame}"
                           for e in timeline.events]}
    events = {e.id: e for e in timeline.events}
    frames = [facts[i] for i in c.frame_ids if i in facts]
    if c.claim_type == "event" and not frames:
        frames = [f for f in facts.values()
                  if any(events[e].start_frame <= f.frame_id <= events[e].end_frame
                         for e in c.event_ids if e in events)]
    visible = Counter(s for f in frames for s in f.structures
                      if f.visible(s) and s != "background")
    out = {
        "frames": f"{len(frames)} frames, ids {_ranges([f.frame_id for f in frames])}, "
                  f"{frames[0].time_s:.1f}-{frames[-1].time_s:.1f} s" if frames else "none",
        "visible_structures": {
            s: f"{n}/{len(frames)} frames, mean area "
               f"{sum(f.structures[s].area for f in frames if f.visible(s)) / n:.0%} of the frame"
            for s, n in visible.most_common()},
        "phase_estimates": dict(Counter(f.phase for f in frames)),
    }
    if c.event_ids:
        out["events"] = [events[e].model_dump(include={"id", "kind", "start_frame", "end_frame",
                                                       "start_s", "end_s", "detail"})
                         for e in c.event_ids if e in events]
    return out


def _ranges(ids: list[int]) -> str:
    """[0, 1, 2, 5, 6] -> "0-2, 5-6"."""
    out, start = [], None
    for i, x in enumerate(ids):
        if start is None:
            start = x
        if i + 1 == len(ids) or ids[i + 1] != x + 1:
            out.append(f"{start}-{x}" if x != start else str(x))
            start = None
    return ", ".join(out)


def judge(claims: list[Claim], results: list[ClaimResult], facts: list[FrameFacts],
          timeline: Timeline, passages: dict[str, str]) -> tuple[list[ClaimResult], dict | None]:
    """Judge the claims the verifier passed; returns updated results and a usage row."""
    passed = {r.claim_id for r in results if r.status == "supported"}
    todo = [c for c in claims if c.id in passed]
    if not ENABLED or not todo:
        return results, None
    by_id = {f.frame_id: f for f in facts}
    payload = [{"claim_id": c.id, "claim_type": c.claim_type, "claim": c.text,
                "evidence": evidence(c, by_id, timeline, passages)} for c in todo]
    messages = [{"role": "user", "content": json.dumps(payload, indent=1)}]
    with generation("judge", MODEL, messages, SYSTEM, {"effort": EFFORT}) as record:
        response = client().messages.parse(
            model=MODEL, max_tokens=8000, system=SYSTEM, messages=messages,
            output_format=Judgement, output_config={"effort": EFFORT},
        )
        record(response)
    if response.parsed_output is None:
        raise RuntimeError(f"judge: no parsed judgement (stop_reason={response.stop_reason})")
    verdicts = {v.claim_id: v for v in response.parsed_output.verdicts}
    out = []
    for r in results:
        v = verdicts.get(r.claim_id)
        if r.status == "supported" and v is not None and v.verdict != "faithful":
            r = r.model_copy(update={"status": "unsupported", "basis": "none",
                                     "reasons": [f"judge: {v.verdict}: {v.reason}"]})
        out.append(r)
    return out, usage_row("judge", response)
