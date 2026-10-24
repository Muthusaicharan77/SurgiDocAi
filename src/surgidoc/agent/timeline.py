"""Node build_timeline: group per-frame facts into phase segments and events."""

from __future__ import annotations

import numpy as np

from .facts import Event, FrameFacts, PhaseSegment, Timeline, validated_classes
from .perception import contiguous_runs

INSTRUMENTS = ("grasper", "l_hook_electrocautery")

# Blood must cover at least 1% of the frame in at least this many consecutive
# sampled frames. The blood class has no test score, so these events are
# always marked unvalidated.
BLEEDING_MIN_AREA = 0.01
BLEEDING_MIN_FRAMES = 2


def _intervals(flags: list[bool], min_len: int = 1) -> list[tuple[int, int]]:
    """(start, end) index pairs, inclusive, of runs of True at least min_len long."""
    out, start = [], None
    for i, f in enumerate([*flags, False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def build_timeline(facts: list[FrameFacts]) -> Timeline:
    if not facts:
        return Timeline(segments=[], events=[], gaps=[])
    times = [f.time_s for f in facts]
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    runs = contiguous_runs(times, max_gap_s=3 * step)
    gaps = [(r.start - 1, r.start) for r in runs[1:]]
    validated = validated_classes()

    segments, events = [], []
    for run in runs:
        chunk = facts[run.start : run.stop]

        # Phase segments: consecutive frames with the same smoothed phase.
        start = 0
        for i in range(1, len(chunk) + 1):
            if i == len(chunk) or chunk[i].phase != chunk[start].phase:
                part = chunk[start:i]
                segments.append(PhaseSegment(
                    phase=part[0].phase, start_frame=part[0].frame_id,
                    end_frame=part[-1].frame_id, start_s=part[0].time_s, end_s=part[-1].time_s,
                    mean_confidence=round(float(np.mean([f.phase_confidence for f in part])), 4),
                ))
                start = i

        for name in INSTRUMENTS:
            for a, b in _intervals([f.visible(name) for f in chunk]):
                events.append(_event("instrument_in_view", chunk[a], chunk[b],
                                     validated=name in validated, detail={"instrument": name}))

        blood = [f.structures["blood"].area if "blood" in f.structures else 0.0 for f in chunk]
        for a, b in _intervals([v >= BLEEDING_MIN_AREA for v in blood], BLEEDING_MIN_FRAMES):
            events.append(_event("possible_bleeding", chunk[a], chunk[b],
                                 validated="blood" in validated,
                                 detail={"max_blood_area": round(max(blood[a : b + 1]), 4)}))

    for i, e in enumerate(sorted(events, key=lambda e: (e.start_frame, e.kind))):
        e.id = f"E{i + 1}"
    events.sort(key=lambda e: int(e.id[1:]))
    return Timeline(segments=segments, events=events, gaps=gaps)


def _event(kind: str, first: FrameFacts, last: FrameFacts, validated: bool,
           detail: dict) -> Event:
    return Event(id="", kind=kind, start_frame=first.frame_id, end_frame=last.frame_id,
                 start_s=first.time_s, end_s=last.time_s, validated=validated, detail=detail)
