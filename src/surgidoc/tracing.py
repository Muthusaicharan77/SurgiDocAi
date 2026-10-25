"""Optional Langfuse tracing.

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (the environment or
.env) and the langfuse package is installed (`pip install -e ".[trace]"`).
LANGFUSE_HOST picks the server: Langfuse Cloud or a self-hosted instance.

One run of the graph becomes one trace: a root span for the run, a span per graph
node (LangGraph callback), and a generation per Claude call with model, token usage,
input and output. Image data is replaced by a placeholder before it is sent.
The run's thread id is the Langfuse session id, so a run and its later review are
grouped together.
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from functools import cache
from pathlib import Path


@cache
def enabled() -> bool:
    from dotenv import load_dotenv

    load_dotenv(Path(".env"))
    return (importlib.util.find_spec("langfuse") is not None
            and bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
            and bool(os.environ.get("LANGFUSE_SECRET_KEY")))


def make_client(**kwargs):
    """A Langfuse client that exports only spans made through Langfuse.

    The Anthropic SDK has its own OpenTelemetry spans, which Langfuse would export as
    a second generation per Claude call, counting every call's tokens twice. The
    generations made here carry the same model and usage, plus the input and output.
    """
    from langfuse import Langfuse
    from langfuse.span_filter import is_langfuse_span

    return Langfuse(should_export_span=is_langfuse_span, **kwargs)


@cache
def _client():
    """The Langfuse client for the configured keys (LANGFUSE_* variables).

    Created explicitly, once: the LangGraph callback looks the client up by public key,
    which only finds a client that already exists.
    """
    enabled()  # loads .env, where the keys usually are
    return make_client()


def strip_images(value):
    """A JSON-able copy of a message list with base64 image data removed."""
    if hasattr(value, "model_dump"):  # SDK content blocks in assistant turns
        value = value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "source": "<jpeg omitted>"}
        return {k: strip_images(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [strip_images(v) for v in value]
    return value


class _Run:
    """Handle for a traced run: graph callbacks, and scores attached to the trace."""

    def __init__(self, trace_id: str | None, callbacks: list) -> None:
        self.trace_id, self.callbacks = trace_id, callbacks

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        if self.trace_id:
            _client().create_score(name=name, value=value, trace_id=self.trace_id,
                                   comment=comment)

    @property
    def url(self) -> str | None:
        return _client().get_trace_url(trace_id=self.trace_id) if self.trace_id else None


@contextmanager
def run_trace(name: str, session_id: str, input: dict, metadata: dict | None = None):
    """Trace one graph invocation. Yields a _Run; a no-op when tracing is off."""
    if not enabled():
        yield _Run(None, [])
        return
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    client = _client()
    with (propagate_attributes(session_id=session_id, trace_name=name, metadata=metadata),
          client.start_as_current_observation(name=name, as_type="agent", input=input) as root):
        handler = CallbackHandler(public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"))
        run = _Run(client.get_current_trace_id(), [handler])
        try:
            yield run
        finally:
            root.update(output={"trace_id": run.trace_id})
    client.flush()


@contextmanager
def generation(name: str, model: str, messages: list, system: str | None = None,
               parameters: dict | None = None):
    """Trace one Claude call. Yields record(response); a no-op when tracing is off."""
    if not enabled():
        yield lambda response: None
        return
    client = _client()
    with client.start_as_current_observation(
        name=name, as_type="generation", model=model, model_parameters=parameters,
        input={"system": system, "messages": strip_images(messages)},
    ) as gen:
        def record(response) -> None:
            u = response.usage
            gen.update(output=strip_images(response.content),
                       usage_details={"input": u.input_tokens, "output": u.output_tokens},
                       metadata={"stop_reason": response.stop_reason})
        yield record
