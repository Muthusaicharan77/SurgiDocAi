"""What the vision models report: per-frame facts and the timeline built from them.

These are the only facts the report may state. The verifier checks every claim
against them, so they must be plain data with no interpretation added.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

# A class counts as visible in a frame if it covers at least this share of the
# frame (0.5%, about 570 px at 256x448). Below that, single stray pixels would count.
MIN_VISIBLE_AREA = 0.005

DEFAULT_SEG_REPORT = Path("reports/seg_test.json")


@cache
def validated_classes(seg_report: Path = DEFAULT_SEG_REPORT) -> frozenset[str]:
    """Classes with a measured test score (RESULTS.md), excluding the black border.

    Read from the evaluation report so retraining updates it automatically.
    Everything else the U-Net predicts (cystic duct, blood, ...) is unvalidated.
    """
    scored = json.loads(Path(seg_report).read_text())["scored_classes"]
    return frozenset(c for c in scored if c != "background")


class StructureFacts(BaseModel):
    area: float  # share of the frame's pixels, 0-1
    components: int  # connected regions of this class


class FrameFacts(BaseModel):
    frame_id: int  # position in the sampled sequence; what claims refer to
    frame_number: int  # frame index in the original 25 fps video
    time_s: float
    source: str  # image path
    phase: str  # smoothed phase prediction
    phase_confidence: float
    structures: dict[str, StructureFacts]  # classes present with area > 0

    def visible(self, name: str) -> bool:
        s = self.structures.get(name)
        return s is not None and s.area >= MIN_VISIBLE_AREA


class PhaseSegment(BaseModel):
    phase: str
    start_frame: int
    end_frame: int  # inclusive frame_id
    start_s: float
    end_s: float
    mean_confidence: float


class Event(BaseModel):
    id: str
    kind: Literal["instrument_in_view", "possible_bleeding"]
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    validated: bool  # False if it rests on a class without a measured test score
    detail: dict[str, str | float]


class Timeline(BaseModel):
    segments: list[PhaseSegment]
    events: list[Event]
    gaps: list[tuple[int, int]]  # (last frame before, first frame after) footage breaks
