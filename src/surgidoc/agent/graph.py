"""The LangGraph pipeline that turns a video folder into a reviewed report.

    sample_frames -> perceive -> build_timeline -> select_key_frames -> retrieve
      -> draft_report -> verify --(failures, retries left)--> draft_report
                              \\--(passed or out of retries)--> drop_unsupported
      -> human_review (pauses the run) -> finalize

    python -m surgidoc.agent.graph run --source /path/to/cholecseg8k/video37
    python -m surgidoc.agent.graph review --thread <id> --approve [--remove c3 c7] [--notes "..."]

State is checkpointed in SQLite after every node, so a run paused at human_review
can be resumed later, from another process.
"""

from __future__ import annotations

import argparse
import json
import operator
import uuid
from functools import cache
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from surgidoc.retrieval.chunking import Passage

from .claims import Claim, ClaimResult
from .draft import draft_report
from .facts import FrameFacts, Timeline
from .perception import Perceiver, sample_frames
from .render import render
from .retrieve import retrieve
from .select import KeyFrame, select_key_frames
from .timeline import build_timeline
from .verify import verify

SEG_WEIGHTS = Path("weights/seg/unet_r34_best.pt")
PHASE_WEIGHTS = Path("weights/phase/transsvnet_resnet50.pth")
CHECKPOINT_DB = Path("data/runs.sqlite")
RUNS_DIR = Path("runs")
MAX_DRAFTS = 3  # first draft + 2 retries


class RunState(TypedDict, total=False):
    # Plain JSON-able values only, so the checkpointer can store every step.
    source: str
    every_n: int
    frames: list[dict]
    facts: list[dict]
    timeline: dict
    key_frames: list[dict]
    selection_fallback: bool
    passages: list[dict]
    queries: list[str]
    claims: list[dict]
    results: list[dict]
    drafts: int
    dropped: list[dict]
    review: dict
    report_path: str
    # Reducer: each node returns only its own usage rows; LangGraph appends them.
    usage: Annotated[list[dict], operator.add]


@cache
def perceiver() -> Perceiver:
    return Perceiver(SEG_WEIGHTS, PHASE_WEIGHTS)


def _facts(s: RunState) -> list[FrameFacts]:
    return [FrameFacts(**f) for f in s["facts"]]


def _timeline(s: RunState) -> Timeline:
    return Timeline(**s["timeline"])


def _claims(s: RunState) -> list[Claim]:
    return [Claim(**c) for c in s.get("claims", [])]


def _passages(s: RunState) -> list[Passage]:
    return [Passage(**p) for p in s["passages"]]


# --- nodes -------------------------------------------------------------------------

def node_sample(s: RunState) -> dict:
    refs = sample_frames(Path(s["source"]), s.get("every_n", 5))
    return {"frames": [{"frame_number": r.frame_number, "path": str(r.path)} for r in refs]}


def node_perceive(s: RunState) -> dict:
    from .perception import FrameRef

    refs = [FrameRef(f["frame_number"], Path(f["path"])) for f in s["frames"]]
    return {"facts": [f.model_dump() for f in perceiver().run(refs)]}


def node_timeline(s: RunState) -> dict:
    return {"timeline": build_timeline(_facts(s)).model_dump()}


def node_select(s: RunState) -> dict:
    keys, info = select_key_frames(_facts(s), _timeline(s))
    return {"key_frames": [k.model_dump() for k in keys],
            "selection_fallback": info["fallback"], "usage": info["usage"]}


def node_retrieve(s: RunState) -> dict:
    passages, queries = retrieve(_timeline(s))
    return {"passages": [p.model_dump() for p in passages], "queries": queries}


def node_draft(s: RunState) -> dict:
    failures = [ClaimResult(**r) for r in s.get("results", []) if r["status"] != "supported"]
    claims, usage = draft_report(
        _facts(s), _timeline(s), [KeyFrame(**k) for k in s["key_frames"]], _passages(s),
        previous=_claims(s) or None, failures=failures or None)
    return {"claims": [c.model_dump() for c in claims], "drafts": s.get("drafts", 0) + 1,
            "usage": [usage]}


def node_verify(s: RunState) -> dict:
    report = verify(_claims(s), _facts(s), _timeline(s), {p["id"] for p in s["passages"]})
    return {"results": [r.model_dump() for r in report.results]}


def route_after_verify(s: RunState) -> str:
    passed = all(r["status"] == "supported" for r in s["results"])
    if passed or s["drafts"] >= MAX_DRAFTS:
        return "drop_unsupported"
    return "draft_report"


def node_drop(s: RunState) -> dict:
    """Out of retries: keep only supported claims and record the rest for the report."""
    results = {r["claim_id"]: r for r in s["results"]}
    keep, dropped = [], []
    for c in s["claims"]:
        r = results[c["id"]]
        if r["status"] == "supported":
            keep.append(c)
        else:
            dropped.append({"claim": c, "status": r["status"], "reasons": r["reasons"]})
    return {"claims": keep, "dropped": dropped}


def node_review(s: RunState) -> dict:
    results = {r["claim_id"]: r for r in s["results"]}
    decision = interrupt({
        "claims": [{**c, "basis": results[c["id"]]["basis"]} for c in s["claims"]],
        "dropped": s.get("dropped", []),
        "instructions": "Resume with {'approve': bool, 'remove': [claim ids], 'notes': str}",
    })
    return {"review": decision}


def route_after_review(s: RunState) -> str:
    return "finalize" if s["review"].get("approve") else END


def node_finalize(s: RunState, config: RunnableConfig) -> dict:
    remove = set(s["review"].get("remove", []))
    claims = [c for c in _claims(s) if c.id not in remove]
    results = {r["claim_id"]: ClaimResult(**r) for r in s["results"]}
    facts = _facts(s)
    report = render(s["source"], facts, _timeline(s), [KeyFrame(**k) for k in s["key_frames"]],
                    claims, results, _passages(s), s.get("dropped", []), s["review"],
                    s.get("usage", []))
    out = RUNS_DIR / config["configurable"]["thread_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report)
    (out / "state.json").write_text(json.dumps({k: v for k, v in s.items() if k != "facts"},
                                               indent=1, default=str))
    return {"report_path": str(out / "report.md")}


def build_graph(checkpointer):
    g = StateGraph(RunState)
    g.add_node("sample_frames", node_sample)
    g.add_node("perceive", node_perceive)
    g.add_node("build_timeline", node_timeline)
    g.add_node("select_key_frames", node_select)
    g.add_node("retrieve", node_retrieve)
    g.add_node("draft_report", node_draft)
    g.add_node("verify", node_verify)
    g.add_node("drop_unsupported", node_drop)
    g.add_node("human_review", node_review)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "sample_frames")
    g.add_edge("sample_frames", "perceive")
    g.add_edge("perceive", "build_timeline")
    g.add_edge("build_timeline", "select_key_frames")
    g.add_edge("select_key_frames", "retrieve")
    g.add_edge("retrieve", "draft_report")
    g.add_edge("draft_report", "verify")
    g.add_conditional_edges("verify", route_after_verify, ["draft_report", "drop_unsupported"])
    g.add_edge("drop_unsupported", "human_review")
    g.add_conditional_edges("human_review", route_after_review, ["finalize", END])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


# --- CLI ---------------------------------------------------------------------------

def _print_pending(state) -> None:
    for task in state.tasks:
        for intr in task.interrupts:
            payload = intr.value
            print(f"\nAwaiting review: {len(payload['claims'])} verified claims, "
                  f"{len(payload['dropped'])} dropped by the verifier.")
            for c in payload["claims"]:
                tag = "" if c["basis"] in ("validated_model", "retrieved_source") else \
                    f" [{c['basis']}]"
                print(f"  {c['id']:4s} {c['claim_type']:18s} {c['text']}{tag}")
            for d in payload["dropped"]:
                print(f"  DROPPED {d['claim']['id']}: {d['claim']['text']} ({'; '.join(d['reasons'])})")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", required=True)
    r.add_argument("--every-n", type=int, default=5)
    r.add_argument("--thread")
    v = sub.add_parser("review")
    v.add_argument("--thread", required=True)
    v.add_argument("--approve", action="store_true")
    v.add_argument("--remove", nargs="*", default=[])
    v.add_argument("--notes", default="")
    args = p.parse_args()

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_graph(saver)
        if args.cmd == "run":
            thread = args.thread or uuid.uuid4().hex[:8]
            config = {"configurable": {"thread_id": thread}}
            for update in graph.stream({"source": args.source, "every_n": args.every_n},
                                       config, stream_mode="updates"):
                for node in update:
                    if node != "__interrupt__":
                        print(f"[{thread}] {node} done")
            _print_pending(graph.get_state(config))
            print(f"\nthread: {thread}")
        else:
            config = {"configurable": {"thread_id": args.thread}}
            result = graph.invoke(Command(resume={"approve": args.approve, "remove": args.remove,
                                                  "notes": args.notes}), config)
            print(result.get("report_path", "not approved; no report written"))


if __name__ == "__main__":
    main()
