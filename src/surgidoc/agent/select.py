"""Node select_key_frames: Claude picks representative frames, using tools to inspect them.

This is the one genuinely agentic step: Claude decides which frames to look at,
looks, and decides again. It is bounded (turns and images) so cost stays fixed,
and its output is checked; if it fails, a deterministic selection is used instead.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from .facts import FrameFacts, Timeline, validated_classes
from .llm import EFFORT, MODEL, client, image_block, usage_row

MAX_TURNS = 6
MAX_IMAGES = 12
MIN_KEY, MAX_KEY = 3, 10

SYSTEM = f"""You select key frames from a laparoscopic cholecystectomy video for a \
documentation report. The frames were analysed by a segmentation model and a phase \
classifier; you get the resulting timeline.

Pick {MIN_KEY} to {MAX_KEY} frames that together document the footage: at least one per \
phase segment, the start of each event, and frames where several structures are clearly \
visible. Avoid near-duplicates. Inspect candidates with the tools before deciding; you can \
view at most {MAX_IMAGES} images in total. When done, answer with the selection and one \
short reason per frame. This is a research demo on public data, not clinical use."""

TOOLS = [
    {
        "name": "frame_stats",
        "description": "Per-frame model output: time, phase and confidence, and visible "
                       "structures with their share of the frame. Unvalidated classes are "
                       "marked; their detections have no measured accuracy.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"frame_ids": {"type": "array", "items": {"type": "integer"}}},
            "required": ["frame_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "view_frames",
        "description": "Return the images of up to 4 frames.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"frame_ids": {"type": "array", "items": {"type": "integer"}}},
            "required": ["frame_ids"],
            "additionalProperties": False,
        },
    },
]


class KeyFrame(BaseModel):
    frame_id: int
    reason: str


class Selection(BaseModel):
    key_frames: list[KeyFrame] = Field(description=f"{MIN_KEY} to {MAX_KEY} frames")


def frame_summary(f: FrameFacts) -> dict:
    validated = validated_classes()
    return {
        "frame_id": f.frame_id, "time_s": f.time_s, "phase": f.phase,
        "phase_confidence": f.phase_confidence,
        "visible": {name: {"area": s.area, "validated": name in validated}
                    for name, s in f.structures.items() if f.visible(name) and name != "background"},
    }


def fallback_selection(facts: list[FrameFacts], timeline: Timeline) -> list[KeyFrame]:
    """Middle frame of each phase segment plus the first frame of each event."""
    picks = {(s.start_frame + s.end_frame) // 2: f"middle of {s.phase} segment"
             for s in timeline.segments}
    for e in timeline.events:
        picks.setdefault(e.start_frame, f"start of {e.kind} ({e.id})")
    return [KeyFrame(frame_id=i, reason=r) for i, r in sorted(picks.items())][:MAX_KEY]


def select_key_frames(facts: list[FrameFacts], timeline: Timeline) -> tuple[list[KeyFrame], dict]:
    """Returns (key frames, info) where info has usage rows and whether the fallback ran."""
    by_id = {f.frame_id: f for f in facts}
    images_left = MAX_IMAGES
    usage = []

    def run_tool(name: str, args: dict) -> list[dict] | str:
        nonlocal images_left
        ids = [i for i in args["frame_ids"] if i in by_id]
        unknown = [i for i in args["frame_ids"] if i not in by_id]
        note = f"Unknown frame ids ignored: {unknown}. " if unknown else ""
        if name == "frame_stats":
            return note + json.dumps([frame_summary(by_id[i]) for i in ids[:40]])
        ids = ids[: min(4, images_left)]
        images_left -= len(ids)
        blocks: list[dict] = []
        for i in ids:
            blocks += [{"type": "text", "text": f"Frame {i} (t={by_id[i].time_s:.1f}s)"},
                       image_block(by_id[i].source)]
        blocks.append({"type": "text", "text": f"{note}{images_left} image views left."})
        return blocks

    messages = [{"role": "user", "content": (
        f"{len(facts)} frames, ids 0-{len(facts) - 1}.\nTimeline:\n"
        f"{timeline.model_dump_json(indent=1)}")}]
    for _ in range(MAX_TURNS):
        response = client().messages.parse(
            model=MODEL, max_tokens=8000, system=SYSTEM, tools=TOOLS, messages=messages,
            output_format=Selection, output_config={"effort": EFFORT},
        )
        usage.append(usage_row("select_key_frames", response))
        if response.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
            for b in response.content if b.type == "tool_use"
        ]})

    selection = response.parsed_output if response.stop_reason == "end_turn" else None
    keys = []
    if selection is not None:
        seen = set()
        for k in selection.key_frames:
            if k.frame_id in by_id and k.frame_id not in seen:
                seen.add(k.frame_id)
                keys.append(k)
    if not MIN_KEY <= len(keys) <= MAX_KEY:
        return fallback_selection(facts, timeline), {"usage": usage, "fallback": True}
    return sorted(keys, key=lambda k: k.frame_id), {"usage": usage, "fallback": False}
