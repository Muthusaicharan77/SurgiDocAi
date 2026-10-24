"""Node finalize: render the approved, verified claims as a Markdown report.

Prose is produced only here, from claims that passed verification and review.
Labels come from each claim's evidence basis, not from the model's wording.
"""

from __future__ import annotations

from surgidoc.retrieval.chunking import Passage

from .claims import Claim, ClaimResult
from .facts import FrameFacts, Timeline
from .select import KeyFrame

LABELS = {
    "validated_model": "",
    "unvalidated_model": " _(unverified: model output without measured accuracy)_",
    "retrieved_source": "",
    "none": "",
}
SECTIONS = [
    ("Observations", ("structures_visible", "phase")),
    ("Events", ("event",)),
    ("Guidance", ("guideline",)),
    ("Limitations", ("limitation",)),
]


def _frames(c: Claim, facts: list[FrameFacts]) -> str:
    if not c.frame_ids:
        return ""
    ids = sorted(c.frame_ids)
    times = f"{facts[ids[0]].time_s:.1f}-{facts[ids[-1]].time_s:.1f} s"
    span = f"frames {ids[0]}-{ids[-1]}" if len(ids) > 2 else f"frames {', '.join(map(str, ids))}"
    return f" [{span}, {times}]"


def render(source: str, facts: list[FrameFacts], timeline: Timeline, key_frames: list[KeyFrame],
           claims: list[Claim], results: dict[str, ClaimResult], passages: list[Passage],
           dropped: list[dict], review: dict, usage: list[dict]) -> str:
    by_passage = {p.id: p for p in passages}
    lines = [
        "# Surgical video documentation (draft)",
        "",
        ("> Research demo on public data (CholecSeg8k). **Not for clinical use.** Every statement "
         "below was checked by code against the vision models' output and approved by a reviewer."),
        "",
        f"- Source: `{source}`",
        f"- Frames analysed: {len(facts)}, footage {facts[0].time_s:.1f}-{facts[-1].time_s:.1f} s"
        + (f" with {len(timeline.gaps)} gap(s)" if timeline.gaps else ""),
        f"- Reviewer decision: {'approved' if review.get('approve') else 'not approved'}",
        "",
        "## Phase timeline (model estimate)",
        "",
        "| Phase | Frames | Time | Mean confidence |",
        "|---|---|---|---|",
        *[f"| {s.phase.replace('_', ' ')} | {s.start_frame}-{s.end_frame} | "
          f"{s.start_s:.1f}-{s.end_s:.1f} s | {s.mean_confidence:.2f} |" for s in timeline.segments],
        "",
        "## Key frames",
        "",
        *[f"- Frame {k.frame_id} ({facts[k.frame_id].time_s:.1f} s): {k.reason} "
          f"(`{facts[k.frame_id].source}`)" for k in key_frames],
    ]
    for title, types in SECTIONS:
        items = [c for c in claims if c.claim_type in types]
        if not items:
            continue
        lines += ["", f"## {title}", ""]
        for c in items:
            label = LABELS[results[c.id].basis]
            if c.claim_type == "phase":
                label = " _(phase model estimate)_"
            cites = "".join(
                f" [{pid}: {by_passage[pid].doc_title[:70]}, {by_passage[pid].section[:50]}]"
                for pid in c.passage_ids)
            lines.append(f"- {c.text}{_frames(c, facts)}{label}{cites}")
            if c.quote:
                lines.append(f"  > \"{c.quote}\"")

    if review.get("notes"):
        lines += ["", "## Reviewer notes", "", review["notes"]]

    removed = review.get("remove", [])
    tokens_in = sum(u["input_tokens"] for u in usage)
    tokens_out = sum(u["output_tokens"] for u in usage)
    lines += [
        "", "## Verification summary", "",
        f"- Claims in report: {len(claims)}",
        f"- Removed by the verifier after the last retry: {len(dropped)}",
        f"- Removed by the reviewer: {len(removed)}",
        f"- Claude calls: {len(usage)}, tokens in/out: {tokens_in}/{tokens_out}",
    ]
    for d in dropped:
        lines.append(f"  - dropped {d['claim']['id']} ({d['status']}): {d['claim']['text']} "
                     f"- {'; '.join(d['reasons'])}")
    return "\n".join(lines) + "\n"
