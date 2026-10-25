"""Tool-use eval: how well does Claude use its tools to pick key frames?

Key-frame selection is the one node where Claude calls tools (frame_stats, view_frames)
and decides for itself what to inspect, so there is no single correct call sequence to
compare against. The eval checks what can be checked instead, on the facts of each
checkpointed run, repeated because the model is not deterministic:

- tool calls:  valid frame ids, image budget, repeated views, whether chosen frames
               were inspected first
- selection:   the rules in the node's prompt (3-10 frames, every phase segment, the
               start of every event, no near-duplicates) and whether the fallback ran
- value:       structures visible in the chosen frames according to the ground-truth
               masks, against the deterministic fallback rule and random frames

    python -m surgidoc.evals.tool_use --threads eval-video17 eval-video37 eval-video55 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from itertools import pairwise
from pathlib import Path
from statistics import mean

from surgidoc.agent.facts import FrameFacts, Timeline, validated_classes
from surgidoc.agent.select import (
    MAX_IMAGES,
    MAX_KEY,
    MIN_KEY,
    fallback_selection,
    select_key_frames,
)

from .grounding import ground_truth_facts, load_run

DEFAULT_OUT = Path("reports/eval_tool_use.json")
EVENT_START_SLACK = 2  # a key frame within 2 sampled frames (0.4 s) of an event start counts
NEAR_DUPLICATE = 2  # key frames at most 2 sampled frames apart, with no footage gap between
RANDOM_DRAWS = 50


def selection_metrics(keys: list[int], timeline: Timeline, gt: list[FrameFacts]) -> dict:
    keys = sorted(keys)
    gap_after = {a for a, _ in timeline.gaps}
    segments = timeline.segments
    events = timeline.events
    near = sum(b - a <= NEAR_DUPLICATE and not any(a <= g < b for g in gap_after)
               for a, b in pairwise(keys))
    validated = validated_classes()
    return {
        "key_frames": len(keys),
        "count_in_range": MIN_KEY <= len(keys) <= MAX_KEY,
        "segment_coverage": mean(any(s.start_frame <= k <= s.end_frame for k in keys)
                                 for s in segments),
        "event_start_coverage": mean(
            any(e.start_frame <= k <= min(e.end_frame, e.start_frame + EVENT_START_SLACK)
                for k in keys) for e in events) if events else None,
        "near_duplicate_pairs": near,
        # Validated structures the masks show in the chosen frames: a proxy for how much
        # each frame documents.
        "structures_per_frame": mean(
            sum(gt[k].visible(s) for s in validated) for k in keys),
    }


def tool_metrics(calls: list[dict], keys: list[int]) -> dict:
    viewed = Counter(i for c in calls if c["tool"] == "view_frames" for i in c["returned"])
    inspected = {i for c in calls for i in c["returned"]}
    return {
        "tool_calls": len(calls),
        "frame_stats_calls": sum(c["tool"] == "frame_stats" for c in calls),
        "view_frames_calls": sum(c["tool"] == "view_frames" for c in calls),
        "calls_with_unknown_ids": sum(bool(c["unknown_ids"]) for c in calls),
        "images_viewed": sum(viewed.values()),
        "within_image_budget": sum(viewed.values()) <= MAX_IMAGES,
        "frames_viewed_twice": sum(n > 1 for n in viewed.values()),
        "key_frames_inspected": mean(k in inspected for k in keys) if keys else 0.0,
    }


def averaged(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows if r[key] is not None]
        out[key] = mean(float(v) for v in values) if values else None
    return out


def main() -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver

    from surgidoc.agent.graph import CHECKPOINT_DB

    p = argparse.ArgumentParser()
    p.add_argument("--threads", nargs="+", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        runs = {t: load_run(saver, t) for t in args.threads}

    rng = random.Random(args.seed)
    agent, fallback, rand, usage = [], [], [], []
    for thread, run in runs.items():
        facts, timeline = run["facts"], run["timeline"]
        gt = ground_truth_facts(facts)

        keys = [k.frame_id for k in fallback_selection(facts, timeline)]
        fallback.append({"thread": thread, "keys": keys,
                         **selection_metrics(keys, timeline, gt)})

        for r in range(args.repeats):
            selected, info = select_key_frames(facts, timeline)
            keys = [k.frame_id for k in selected]
            usage += info["usage"]
            agent.append({"thread": thread, "repeat": r, "keys": keys,
                          "fallback": info["fallback"], "turns": info["turns"],
                          "reasons": [k.reason for k in selected],
                          "calls": info["tool_calls"],
                          **selection_metrics(keys, timeline, gt),
                          **tool_metrics(info["tool_calls"], keys)})
            print(f"{thread} repeat {r}: {len(keys)} key frames, "
                  f"{len(info['tool_calls'])} tool calls, fallback={info['fallback']}")

            # Random frames, as many as the agent picked this time.
            draws = [selection_metrics(rng.sample(range(len(facts)), len(keys)), timeline, gt)
                     for _ in range(RANDOM_DRAWS)]
            rand.append({"thread": thread, **averaged(draws)})

    strip = ("thread", "repeat", "keys", "reasons", "calls")
    summary = {
        "agent": averaged([{k: v for k, v in r.items() if k not in strip} for r in agent]),
        "fallback_rule": averaged([{k: v for k, v in r.items() if k not in strip}
                                   for r in fallback]),
        "random_frames": averaged([{k: v for k, v in r.items() if k not in strip}
                                   for r in rand]),
    }
    report = {"threads": args.threads, "repeats": args.repeats, "summary": summary,
              "agent": agent, "fallback_rule": fallback, "usage": usage,
              "tokens_in": sum(u["input_tokens"] for u in usage),
              "tokens_out": sum(u["output_tokens"] for u in usage)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    sel = ["key_frames", "count_in_range", "segment_coverage", "event_start_coverage",
           "near_duplicate_pairs", "structures_per_frame"]
    print("\n| Selection | " + " | ".join(sel) + " |")
    print("|---|" + "---:|" * len(sel))
    for name, s in summary.items():
        print(f"| {name} | " + " | ".join(f"{s[m]:.2f}" for m in sel) + " |")
    tools = ["fallback", "turns", "tool_calls", "frame_stats_calls", "view_frames_calls",
             "calls_with_unknown_ids", "images_viewed", "within_image_budget",
             "frames_viewed_twice", "key_frames_inspected"]
    print("\nAgent tool use, mean per selection:")
    for m in tools:
        print(f"  {m}: {summary['agent'][m]:.2f}")
    print(f"\ntokens in/out: {report['tokens_in']}/{report['tokens_out']}; wrote {args.out}")


if __name__ == "__main__":
    main()
