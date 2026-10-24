"""Grounding eval: are the report's claims true according to the hand-drawn masks?

The verifier checks claims against the vision models' output. This eval checks the
same claims against CholecSeg8k's ground-truth masks instead, by rebuilding each
frame's facts from its mask and running the same verifier on them. A claim the
verifier passes but the masks contradict is a vision error that reached the report.

Two conditions per run:
- no_verifier:   Claude's first draft, as written
- with_verifier: the claims that survived verification and retries (what the reviewer sees)

Only claims about structures can be graded: structures_visible claims, and event
claims that name structures. A claim is graded on the structures it lists and the
ones its text names, so a first-draft claim the verifier rejected for leaving a named
structure unlisted is still graded on whether it is true. Phase claims have no ground truth here (no Cholec80
phase labels), and guideline and limitation claims make no observation.

    python -m surgidoc.agent.graph run --source /path/to/cholecseg8k/video17 --thread eval-video17
    python -m surgidoc.evals.grounding --threads eval-video17 eval-video37 eval-video55
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from surgidoc.agent.claims import Claim
from surgidoc.agent.facts import FrameFacts, StructureFacts, Timeline, validated_classes
from surgidoc.agent.verify import mentioned_structures, verify
from surgidoc.vision.classes import CLASS_NAMES, IGNORE_INDEX, encode_mask
from surgidoc.vision.dataset import IMAGE_SUFFIX, MASK_SUFFIX

DEFAULT_OUT = Path("reports/eval_grounding.json")
GRADABLE = ("structures_visible", "event")


def ground_truth_facts(facts: list[FrameFacts]) -> list[FrameFacts]:
    """The same frames, with structures measured on the ground-truth mask.

    Areas are shares of all pixels, as for the model's masks; unlabeled pixels
    count in the denominator but belong to no class. Phase is copied unchanged,
    since there is no phase ground truth.
    """
    out = []
    for f in facts:
        mask = encode_mask(np.array(Image.open(f.source.replace(IMAGE_SUFFIX, MASK_SUFFIX))))
        structures = {}
        for c in np.unique(mask):
            if c == IGNORE_INDEX:
                continue
            region = mask == c
            structures[CLASS_NAMES[c]] = StructureFacts(
                area=round(float(region.mean()), 5), components=int(ndimage.label(region)[1]))
        out.append(f.model_copy(update={"structures": structures}))
    return out


def named_structures(c: Claim) -> list[str]:
    return sorted(set(c.structures) | mentioned_structures(c.text))


def gradable(c: Claim) -> bool:
    return c.claim_type in GRADABLE and bool(named_structures(c))


def grade(claims: list[Claim], facts: list[FrameFacts], gt_facts: list[FrameFacts],
          timeline: Timeline, passages: dict[str, str]) -> list[dict]:
    """Per claim: the verifier's verdict on model output, and on the ground truth."""
    model = {r.claim_id: r for r in verify(claims, facts, timeline, passages).results}
    # Against the masks, grade every structure the claim names. Left invalid after
    # that are claims citing frames or events that do not exist.
    complete = [c.model_copy(update={"structures": named_structures(c)}) for c in claims]
    truth = {r.claim_id: r for r in verify(complete, gt_facts, timeline, passages).results}
    rows = []
    for c in claims:
        row = {"id": c.id, "claim_type": c.claim_type, "text": c.text,
               "structures": named_structures(c), "model_status": model[c.id].status,
               "gradable": gradable(c)}
        if row["gradable"]:
            row["gt_status"] = truth[c.id].status
            row["gt_reasons"] = truth[c.id].reasons
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["gradable"]]
    n = len(graded)
    counts = Counter(r["gt_status"] for r in graded)
    wrong = Counter(s for r in graded if r["gt_status"] == "unsupported"
                    for s in r["structures"]
                    if any(reason.startswith(f"{s} ") for reason in r["gt_reasons"]))
    validated = validated_classes()
    return {
        "claims": len(rows),
        "graded": n,
        "grounded": counts["supported"],
        "contradicted": counts["unsupported"],
        "invalid": counts["invalid"],
        "grounded_rate": counts["supported"] / n if n else None,
        "contradicted_rate": counts["unsupported"] / n if n else None,
        # Claims that look fine against the model's output but not against the masks.
        "passed_verifier_but_contradicted": sum(
            r["model_status"] == "supported" and r["gt_status"] != "supported" for r in graded),
        "contradicted_structures": dict(wrong),
        # Contradictions on classes without a measured test score are labelled
        # "unverified" in the report; those on validated classes are not.
        "contradicted_on_validated_classes": sum(n for s, n in wrong.items() if s in validated),
    }


def load_run(saver, thread: str) -> dict:
    """Facts, timeline, passages, first-draft claims and final claims of a checkpointed run."""
    from surgidoc.agent.graph import build_graph

    graph = build_graph(saver)
    config = {"configurable": {"thread_id": thread}}
    state = graph.get_state(config)
    if "human_review" not in state.next and "report_path" not in state.values:
        raise ValueError(f"run {thread} has not reached human review")
    first = next((h.values for h in graph.get_state_history(config)
                  if h.next == ("verify",) and h.values.get("drafts") == 1), None)
    if first is None:
        raise ValueError(f"run {thread}: no first draft in the checkpoint history")
    v = state.values
    return {
        "source": v["source"],
        "facts": [FrameFacts(**f) for f in v["facts"]],
        "timeline": Timeline(**v["timeline"]),
        "passages": {p["id"]: p["text"] for p in v["passages"]},
        "drafts": v["drafts"],
        "no_verifier": [Claim(**c) for c in first["claims"]],
        "with_verifier": [Claim(**c) for c in v["claims"]],
    }


def pooled(per_run: list[dict], condition: str) -> dict:
    return summarise([r for run in per_run for r in run[condition]["rows"]])


def main() -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver

    from surgidoc.agent.graph import CHECKPOINT_DB

    p = argparse.ArgumentParser()
    p.add_argument("--threads", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    per_run = []
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        for thread in args.threads:
            run = load_run(saver, thread)
            gt = ground_truth_facts(run["facts"])
            result = {"thread": thread, "source": run["source"], "drafts": run["drafts"]}
            for condition in ("no_verifier", "with_verifier"):
                rows = grade(run[condition], run["facts"], gt, run["timeline"], run["passages"])
                result[condition] = {"summary": summarise(rows), "rows": rows}
            per_run.append(result)

    report = {
        "runs": per_run,
        "pooled": {c: pooled(per_run, c) for c in ("no_verifier", "with_verifier")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    print("| Run | Condition | Claims | Graded | Grounded | Contradicted | Invalid |"
          " Passed verifier, contradicted | Contradicted, validated class |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, results in [(r["thread"], r) for r in per_run] + [("pooled", None)]:
        for condition in ("no_verifier", "with_verifier"):
            s = results[condition]["summary"] if results else report["pooled"][condition]
            print(f"| {name} | {condition} | {s['claims']} | {s['graded']} | {s['grounded']} | "
                  f"{s['contradicted']} | {s['invalid']} | {s['passed_verifier_but_contradicted']} | "
                  f"{s['contradicted_on_validated_classes']} |")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
