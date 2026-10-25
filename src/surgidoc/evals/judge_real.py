"""Judge eval on real claims: does the judge agree with a human reading?

The injected-error eval plants blatant errors; this one uses the claims Claude
actually wrote. Each claim the verifier passes is labelled by hand as faithful or
overstating its evidence (configs/evals/judge_labels.json), then judged, and the
judge is scored against the labels.

The judge's prompt and evidence were adjusted on the test-video runs, so their
numbers are optimistic. The held-out runs (validation videos, drafted after the
adjustment) were labelled before the judge was run on them.

    python -m surgidoc.evals.judge_real --set heldout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from surgidoc.agent.judge import judge
from surgidoc.agent.verify import verify

from .grounding import load_run

DEFAULT_LABELS = Path("configs/evals/judge_labels.json")
DEFAULT_OUT = Path("reports/eval_judge_real.json")


def confusion(rows: list[dict]) -> dict:
    tp = sum(r["label"] != "faithful" and r["judged"] != "faithful" for r in rows)
    fp = sum(r["label"] == "faithful" and r["judged"] != "faithful" for r in rows)
    fn = sum(r["label"] != "faithful" and r["judged"] == "faithful" for r in rows)
    tn = sum(r["label"] == "faithful" and r["judged"] == "faithful" for r in rows)
    return {"claims": len(rows), "overstating_by_label": tp + fn, "caught": tp,
            "missed": fn, "false_alarms": fp, "faithful_passed": tn}


def main() -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver

    from surgidoc.agent.graph import CHECKPOINT_DB

    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, help="a key of the labels file, e.g. heldout")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    spec = json.loads(args.labels.read_text())["sets"][args.set]
    rows = []
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        for thread, labels in spec["runs"].items():
            run = load_run(saver, thread)
            claims = run[spec["claims"]]
            results = verify(claims, run["facts"], run["timeline"], run["passages"]).results
            judged, _ = judge(claims, results, run["facts"], run["timeline"], run["passages"])
            for c, r, j in zip(claims, results, judged):
                if r.status != "supported":
                    continue  # the judge only sees what the verifier passed
                if c.id not in labels:
                    raise ValueError(f"{thread} {c.id} passed the verifier but has no label")
                rows.append({"thread": thread, "id": c.id, "claim_type": c.claim_type,
                             "text": c.text, "label": labels[c.id],
                             "judged": "faithful" if j.status == "supported"
                             else j.reasons[0].split(":")[1].strip(),
                             "reason": j.reasons[0] if j.reasons else ""})

    summary = confusion(rows)
    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    out[args.set] = {"summary": summary, "rows": rows}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["label"] != "faithful" or r["judged"] != "faithful":
            print(f"  {r['thread']} {r['id']} label={r['label']} judged={r['judged']}: {r['text']}")


if __name__ == "__main__":
    main()
