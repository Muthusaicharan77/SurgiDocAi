import json
from types import SimpleNamespace as NS
from typing import TypedDict

import pytest

from surgidoc import tracing


def test_strip_images_replaces_image_data():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "frame 3"},
        {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
        {"type": "tool_result", "content": [{"type": "image", "source": {"data": "BBBB"}}]},
    ]}]
    out = json.dumps(tracing.strip_images(messages))
    assert "AAAA" not in out and "BBBB" not in out and "frame 3" in out


def test_tracing_is_a_no_op_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(tracing, "enabled", lambda: False)
    with tracing.run_trace("run", "t1", {}) as run:
        assert run.callbacks == [] and run.trace_id is None
        run.score("drafts", 1)
        with tracing.generation("g", "m", []) as record:
            record(None)


def test_graph_run_becomes_one_nested_trace(monkeypatch):
    pytest.importorskip("langfuse")
    from langfuse import Langfuse
    from langgraph.graph import END, START, StateGraph
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(tracing, "enabled", lambda: True)
    exporter = InMemorySpanExporter()
    client = Langfuse(public_key="pk-test", secret_key="sk-test", base_url="http://localhost:1",
                      span_exporter=exporter)
    monkeypatch.setattr(tracing, "_client", lambda: client)

    class State(TypedDict, total=False):
        x: int

    def draft(_):
        messages = [{"role": "user", "content": [{"type": "image", "source": {"data": "AAAA"}}]}]
        with tracing.generation("draft_report", "claude-haiku-5-5", messages, "sys") as record:
            record(NS(usage=NS(input_tokens=10, output_tokens=3), stop_reason="end_turn",
                      content=[{"type": "text", "text": "ok"}]))
        return {"x": 1}

    g = StateGraph(State)
    g.add_node("draft", draft)
    g.add_edge(START, "draft")
    g.add_edge("draft", END)
    graph = g.compile()
    with tracing.run_trace("surgidoc run", "thread-1", {"source": "v"}) as run:
        graph.invoke({}, {"callbacks": run.callbacks})
    client.flush()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert {"surgidoc run", "draft", "draft_report"} <= set(spans)
    assert len({s.context.trace_id for s in spans.values()}) == 1
    root, node, gen = spans["surgidoc run"], spans["draft"], spans["draft_report"]
    assert root.parent is None and gen.parent.span_id == node.context.span_id
    attrs = dict(gen.attributes)
    assert attrs["langfuse.observation.type"] == "generation"
    assert json.loads(attrs["langfuse.observation.usage_details"]) == {"input": 10, "output": 3}
    assert "AAAA" not in attrs["langfuse.observation.input"]
    assert all(dict(s.attributes).get("session.id") == "thread-1" for s in spans.values())
