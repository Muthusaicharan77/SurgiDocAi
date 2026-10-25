"""Node draft_report: Claude turns the evidence into a list of claims.

Claude sees the timeline, the key frames (images plus model output) and the
retrieved passages, and must answer in the Claim schema. On a retry it also gets
its previous draft and the verifier's reasons for each failed claim.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from surgidoc.retrieval.chunking import Passage
from surgidoc.tracing import generation
from surgidoc.vision.classes import CLASS_NAMES
from surgidoc.vision.phase import PHASES

from .claims import Claim, ClaimResult
from .facts import FrameFacts, Timeline, validated_classes
from .llm import EFFORT, MODEL, client, image_block, usage_row
from .select import KeyFrame, frame_summary


class Draft(BaseModel):
    claims: list[Claim]


def system_prompt() -> str:
    validated = sorted(validated_classes())
    unvalidated = sorted(set(CLASS_NAMES) - set(validated) - {"background"})
    return f"""You draft the documentation of a laparoscopic cholecystectomy video as a list \
of claims. Research demo on public data, not clinical use.

Every claim is checked by code against the vision models' output before anyone sees it. \
Claims that cite missing evidence are rejected. Write only what the evidence shows.

Claim types:
- structures_visible: structures visible in frames. Set frame_ids and structures.
- phase: frames belong to a surgical phase. Set frame_ids and phase.
- event: an event from the timeline. Set event_ids (and structures if the text names any).
- guideline: guidance from a retrieved passage. Set passage_ids and quote; no frames, no \
structures. State guidance, never an observation. quote is copied character for character \
from one cited passage, 6-60 words, and must contain the guidance the claim states. Do not \
claim more than the quote says.
- limitation: something the system could not determine. No evidence fields.

Rules:
- One fact per claim, one sentence each. Ids c1, c2, ...
- Structure ids: {", ".join(CLASS_NAMES)}. Phase ids: {", ".join(PHASES)}.
- If the text names a structure (including instruments, blood, bleeding), list it in \
structures, except in guideline and limitation claims.
- frame_ids may be any analysed frame, not only key frames. For a span, list every id in \
it (e.g. 0-15 -> [0, 1, ..., 15]). A claim about several frames must hold in at least 80% \
of them.
- Validated classes (measured test accuracy): {", ".join(validated)}.
- Unvalidated classes (no measured accuracy): {", ".join(unvalidated)}. Claims about them \
are labelled unverified in the report. Phrase bleeding as "possible bleeding".
- The phase classifier has no measured accuracy here; phase claims are labelled estimates.
- Cite only passage ids listed below. Prefer guidance on what should be documented and on \
safety steps. When guidance asks for something the system cannot check (e.g. the critical \
view of safety, which needs the cystic duct and artery), add a limitation claim saying the \
surgeon must confirm it.

Cover: the phase sequence, what is visible in the key frames, instrument and bleeding \
events, relevant guidance, and limitations. About 8-20 claims."""


def build_messages(facts: list[FrameFacts], timeline: Timeline, key_frames: list[KeyFrame],
                   passages: list[Passage], previous: list[Claim] | None,
                   failures: list[ClaimResult] | None) -> list[dict]:
    by_id = {f.frame_id: f for f in facts}
    content: list[dict] = [{"type": "text", "text": (
        f"{len(facts)} frames analysed (ids 0-{len(facts) - 1}).\n"
        f"Timeline:\n{timeline.model_dump_json(indent=1)}\n\nKey frames follow.")}]
    for k in key_frames:
        f = by_id[k.frame_id]
        content += [
            {"type": "text", "text": f"Key frame {f.frame_id} ({k.reason}):\n"
                                     f"{json.dumps(frame_summary(f))}"},
            image_block(f.source),
        ]
    content.append({"type": "text", "text": "Retrieved guideline passages:\n\n" + "\n\n".join(
        f"[{p.id}] {p.doc_title} - {p.section}\n{p.text}" for p in passages)})
    if previous is not None and failures:
        content.append({"type": "text", "text": (
            "Your previous draft failed verification. Fix or remove the failed claims and keep "
            "the rest.\nPrevious draft:\n"
            + json.dumps([c.model_dump() for c in previous], indent=1)
            + "\nFailures:\n"
            + "\n".join(f"{r.claim_id}: {r.status}: {'; '.join(r.reasons)}" for r in failures))})
    return [{"role": "user", "content": content}]


def draft_report(facts, timeline, key_frames, passages, previous=None, failures=None):
    system = system_prompt()
    messages = build_messages(facts, timeline, key_frames, passages, previous, failures)
    name = "draft_report" if previous is None else "draft_report retry"
    with generation(name, MODEL, messages, system, {"effort": EFFORT, "max_tokens": 16000}) as record:
        response = client().messages.parse(
            model=MODEL, max_tokens=16000, system=system, messages=messages,
            output_format=Draft, output_config={"effort": EFFORT},
        )
        record(response)
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        raise RuntimeError(f"draft_report: no parsed draft (stop_reason={response.stop_reason})")
    return response.parsed_output.claims, usage_row("draft_report", response)
