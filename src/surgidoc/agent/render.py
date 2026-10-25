"""Node finalize: render the approved, verified claims as a Markdown report.

Prose is produced only here, from claims that passed verification and review.
Labels come from each claim's evidence basis, not from the model's wording.
"""

from __future__ import annotations

from pathlib import Path

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


def frame_spans(ids: list[int], facts: list[FrameFacts]) -> str:
    """Cited frames as runs with times: "frames 0-31 (22.4-28.6 s), 47 (35.0 s)".

    Every cited frame is shown; a run is only drawn where the ids are consecutive, so
    the text never suggests frames that were not cited.
    """
    ids = sorted(set(ids))
    runs, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[i - 1] + 1:
            a, b = ids[start], ids[i - 1]
            runs.append(f"{a} ({facts[a].time_s:.1f} s)" if a == b else
                        f"{a}-{b} ({facts[a].time_s:.1f}-{facts[b].time_s:.1f} s)")
            start = i
    return ("frame " if len(ids) == 1 else "frames ") + ", ".join(runs)


def _evidence_span(c: Claim, facts: list[FrameFacts], timeline: Timeline) -> str:
    if c.frame_ids:
        return f" [{frame_spans(c.frame_ids, facts)}]"
    events = {e.id: e for e in timeline.events}
    spans = [f"{e}: {frame_spans(list(range(events[e].start_frame, events[e].end_frame + 1)), facts)}"
             for e in c.event_ids if e in events]
    return f" [{'; '.join(spans)}]" if spans else ""


def _display_path(path: str, source: str) -> str:
    """Frame paths relative to the source folder, so reports carry no local paths."""
    try:
        return str(Path(path).relative_to(source))
    except ValueError:
        return Path(path).name


def render(source: str, facts: list[FrameFacts], timeline: Timeline, key_frames: list[KeyFrame],
           claims: list[Claim], results: dict[str, ClaimResult], passages: list[Passage],
           dropped: list[dict], review: dict, usage: list[dict]) -> str:
    by_passage = {p.id: p for p in passages}
    lines = [
        "# Surgical video documentation (draft)",
        "",
        ("> Research demo on public data (CholecSeg8k). **Not for clinical use.** Each statement "
         "under Observations, Events, Guidance and Limitations is a claim that was checked by code "
         "against the vision models' output or its quoted guideline passage, and by a second model "
         "for wording that goes beyond that evidence. The phase timeline and key frames are model "
         "output, not checked claims. The reviewer's decision is stated below."),
        "",
        f"- Source: `{Path(source).name}`",
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
        *[f"- Frame {k.frame_id} ({facts[k.frame_id].time_s:.1f} s): "
          f"`{_display_path(facts[k.frame_id].source, source)}`" for k in key_frames],
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
            lines.append(f"- {c.text}{_evidence_span(c, facts, timeline)}{label}{cites}")
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
        f"- Removed by the verifier or judge after the last retry: {len(dropped)}",
        f"- Removed by the reviewer: {len(removed)}",
        f"- Claude calls: {len(usage)}, tokens in/out: {tokens_in}/{tokens_out}",
    ]
    for d in dropped:
        lines.append(f"  - dropped {d['claim']['id']} ({d['status']}): {d['claim']['text']} "
                     f"- {'; '.join(d['reasons'])}")
    return "\n".join(lines) + "\n"
