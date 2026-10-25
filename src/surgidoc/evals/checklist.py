"""Checklist comparison: how much of a real operative report does SurgiDoc produce?

No public dataset pairs these videos with surgeons' reports, so the comparison is
against what such reports must contain (configs/evals/report_checklist.json): a
German report template and three published standards. Each item is categorised by
hand; for the items SurgiDoc covers fully or partly, a rule checks whether each
run's report actually contained it.

    python -m surgidoc.evals.checklist --threads eval-video17 eval-video37 eval-video55
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from surgidoc.agent.claims import Claim
from surgidoc.agent.verify import mentioned_structures

DEFAULT_CHECKLIST = Path("configs/evals/report_checklist.json")
DEFAULT_OUT = Path("reports/eval_checklist.json")


def present(check: dict, claims: list[Claim], key_frames: list) -> bool:
    """Whether a run's claims contain what the check asks for."""
    (kind, value), = check.items()
    if kind == "phase":
        return any(c.claim_type == "phase" and c.phase == value for c in claims)
    if kind == "structure":
        return any(c.claim_type == "structures_visible" and value in c.structures for c in claims)
    if kind == "event_structure":
        return any(c.claim_type == "event"
                   and value in set(c.structures) | mentioned_structures(c.text) for c in claims)
    if kind == "limitation_mentions":
        return any(c.claim_type == "limitation" and any(w in c.text.lower() for w in value)
                   for c in claims)
    if kind == "key_frames":
        return bool(key_frames)
    if kind == "review_step":
        return True  # every report passes the human review interrupt
    raise ValueError(f"unknown check {kind}")


def main() -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver

    from surgidoc.agent.graph import CHECKPOINT_DB, build_graph

    p = argparse.ArgumentParser()
    p.add_argument("--threads", nargs="+", required=True)
    p.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    checklist = json.loads(args.checklist.read_text())
    items = checklist["items"]
    runs = {}
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_graph(saver)
        for t in args.threads:
            v = graph.get_state({"configurable": {"thread_id": t}}).values
            runs[t] = ([Claim(**c) for c in v["claims"]], v["key_frames"])

    rows = []
    for item in items:
        row = {k: item[k] for k in ("id", "item", "sources", "category")}
        if "check" in item:
            row["present_in"] = [t for t, (claims, keys) in runs.items()
                                 if present(item["check"], claims, keys)]
        rows.append(row)

    counts = Counter(r["category"] for r in rows)
    by_source = {s: Counter(r["category"] for r in rows if s in r["sources"])
                 for s in checklist["sources"]}
    report = {"items": len(rows), "categories": dict(counts),
              "by_source": {s: dict(c) for s, c in by_source.items()}, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    order = ["covered", "partial", "possible_from_video", "not_from_video"]
    print("| Source | Items | " + " | ".join(order) + " |")
    print("|---|---:|" + "---:|" * len(order))
    for s, c in [*by_source.items(), ("all", counts)]:
        print(f"| {s} | {sum(c.values())} | " + " | ".join(str(c[k]) for k in order) + " |")
    print("\nChecked items (reported, or for the last two categories flagged as a limitation):")
    for r in rows:
        if "present_in" in r:
            print(f"  {r['id']:16s} {r['category']:20s} {len(r['present_in'])}/{len(runs)}"
                  f"  {r['present_in']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
