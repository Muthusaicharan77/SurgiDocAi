"""Retrieval eval: does search find the passages that answer a question?

Hand-labelled questions (configs/evals/retrieval_questions.json), scored for dense
search alone and for dense search plus each cross-encoder reranker given. Every
ranking reorders the same 25 dense candidates, so the comparison isolates the reranker.

    python -m surgidoc.evals.retrieval --rerankers BAAI/bge-reranker-base other/model

Metrics, averaged over questions:
- hit@k:    a relevant passage is in the top k
- recall@k: share of a question's relevant passages in the top k
- mrr:      1 / rank of the first relevant passage (0 if none in the candidates)
k = 3 is what the agent uses per query (surgidoc.agent.retrieve.PER_QUERY).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_QUESTIONS = Path("configs/evals/retrieval_questions.json")
DEFAULT_OUT = Path("reports/eval_retrieval.json")
KS = (1, 3, 5)
CANDIDATES = 25


def score(ranking: list[str], relevant: set[str]) -> dict:
    first = next((i for i, pid in enumerate(ranking, 1) if pid in relevant), None)
    out = {"mrr": 1 / first if first else 0.0}
    for k in KS:
        top = set(ranking[:k])
        out[f"hit@{k}"] = float(bool(top & relevant))
        out[f"recall@{k}"] = len(top & relevant) / len(relevant)
    return out


def mean(rows: list[dict]) -> dict:
    return {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]}


def main() -> None:
    from sentence_transformers import CrossEncoder

    from surgidoc.retrieval.index import RERANK_MODEL, GuidelineIndex

    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--rerankers", nargs="+", default=[RERANK_MODEL])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    questions = json.loads(args.questions.read_text())["questions"]
    index = GuidelineIndex(rerank=False)
    candidates = {q["id"]: index.search(q["question"], k=CANDIDATES) for q in questions}
    index.client.close()

    rankings = {"dense": {qid: [h.passage.id for h in hits] for qid, hits in candidates.items()}}
    for model in args.rerankers:
        encoder = CrossEncoder(model, device="cpu")
        rankings[model] = {}
        for q in questions:
            hits = candidates[q["id"]]
            scores = encoder.predict([(q["question"], h.passage.text) for h in hits])
            order = sorted(zip(scores, hits), key=lambda pair: pair[0], reverse=True)
            rankings[model][q["id"]] = [h.passage.id for _, h in order]

    rows = []
    for q in questions:
        relevant = set(q["relevant"])
        rows.append({"id": q["id"], "question": q["question"], "relevant": q["relevant"],
                     **{name: {"top5": r[q["id"]][:5], **score(r[q["id"]], relevant)}
                        for name, r in rankings.items()}})
    summary = {name: mean([{m: v for m, v in row[name].items() if m != "top5"} for row in rows])
               for name in rankings}
    report = {"questions": len(rows), "candidates": CANDIDATES, "summary": summary, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    metrics = list(summary["dense"])
    print("| Ranking | " + " | ".join(metrics) + " |")
    print("|---|" + "---:|" * len(metrics))
    for name, values in summary.items():
        label = "dense only" if name == "dense" else f"dense + {name}"
        print(f"| {label} | " + " | ".join(f"{values[m]:.2f}" for m in metrics) + " |")
    print(f"\n{len(rows)} questions, {CANDIDATES} dense candidates each; wrote {args.out}")


if __name__ == "__main__":
    main()
