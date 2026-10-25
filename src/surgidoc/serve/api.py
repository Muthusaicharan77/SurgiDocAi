"""HTTP API: upload a video, follow the run, review its claims, fetch the report.

    uvicorn surgidoc.serve.api:app --port 8000

    POST /runs                  upload a video (multipart field "video"); returns the run id
    GET  /runs/{id}             status, nodes done, and the claims once it awaits review
    GET  /runs/{id}/events      progress as Server-Sent Events, until review or the end
    POST /runs/{id}/review      {"approve": bool, "remove": [claim ids], "notes": str}
    GET  /runs/{id}/report      the Markdown report, once approved

Runs execute one at a time in a background worker: perception is CPU-bound and the
local guideline index admits one process. State lives in the same SQLite checkpoints
the CLI uses, so a run's status survives a restart (progress events do not), and runs
started from the CLI can be reviewed here.

Research demo, not for clinical use. There is no authentication: run it locally or
behind your own access control.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from surgidoc.agent import graph as pipeline

from .frames import extract_frames

UPLOADS_DIR = Path(os.environ.get("SURGIDOC_UPLOADS", "data/uploads"))
MAX_UPLOAD_BYTES = int(os.environ.get("SURGIDOC_MAX_UPLOAD_MB", "2000")) * 2**20
# Lets POST /runs take a server-side frame folder instead of an upload (development).
ALLOW_LOCAL_SOURCE = os.environ.get("SURGIDOC_ALLOW_LOCAL_SOURCE") == "1"
TERMINAL = {"awaiting_review", "done", "failed"}


class Review(BaseModel):
    approve: bool
    remove: list[str] = []
    notes: str = ""


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued, running, awaiting_review, reviewing, done, failed
    events: list[dict] = field(default_factory=list)
    error: str | None = None
    trace_url: str | None = None


class Runner:
    """One worker thread runs the graph; request threads read status and events."""

    def __init__(self, graph) -> None:
        self.graph = graph
        self.jobs: dict[str, Job] = {}
        self.changed = threading.Condition()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="surgidoc-run")

    def _update(self, job: Job, **changes) -> None:
        with self.changed:
            for k, v in changes.items():
                setattr(job, k, v)
            self.changed.notify_all()

    def _event(self, job: Job, kind: str, **data) -> None:
        with self.changed:
            job.events.append({"seq": len(job.events), "kind": kind, "time": time.time(),
                               **data})
            self.changed.notify_all()

    def submit_run(self, run_id: str, video: Path | None, source: Path | None,
                   label: str | None = None) -> Job:
        job = self.jobs[run_id] = Job(run_id)
        self.pool.submit(self._run, job, video, source, label)
        return job

    def _run(self, job: Job, video: Path | None, source: Path | None,
             label: str | None) -> None:
        self._update(job, status="running")
        try:
            if video is not None:
                frames = video.parent / "frames"
                n = extract_frames(video, frames)
                self._event(job, "node", node="extract_frames", frames=n)
                inputs = {"source": str(frames), "every_n": 1}
            else:
                inputs = {"source": str(source), "every_n": 5}
            if label:
                inputs["label"] = label
            state, url = pipeline.start_run(
                self.graph, job.id, inputs, on_node=lambda n: self._event(job, "node", node=n))
            status = "awaiting_review" if pipeline.pending_review(state) else "done"
            self._update(job, status=status, trace_url=url)
            self._event(job, "status", status=status)
        except Exception as e:  # noqa: BLE001 - any failure must reach the client, not just the log
            self._update(job, status="failed", error=f"{type(e).__name__}: {e}")
            self._event(job, "status", status="failed", error=job.error)

    def submit_review(self, run_id: str, review: Review) -> Job:
        job = self.jobs.setdefault(run_id, Job(run_id))
        self._update(job, status="reviewing")
        self.pool.submit(self._review, job, review)
        return job

    def _review(self, job: Job, review: Review) -> None:
        try:
            pipeline.review_run(self.graph, job.id, review.model_dump())
            self._update(job, status="done")
            self._event(job, "status", status="done")
        except Exception as e:  # noqa: BLE001 - as in _run
            self._update(job, status="failed", error=f"{type(e).__name__}: {e}")
            self._event(job, "status", status="failed", error=job.error)

    def checkpoint(self, run_id: str):
        return self.graph.get_state({"configurable": {"thread_id": run_id}})

    def status(self, run_id: str) -> dict:
        job = self.jobs.get(run_id)
        state = self.checkpoint(run_id)
        if job is None and not state.values:
            raise HTTPException(404, f"no run {run_id}")
        if job is not None and job.status in ("queued", "running", "reviewing", "failed"):
            status = job.status
        elif pipeline.pending_review(state) is not None:
            status = "awaiting_review"
        else:
            status = "done"
        out = {"id": run_id, "status": status,
               "nodes_done": [e["node"] for e in (job.events if job else [])
                              if e["kind"] == "node"],
               "error": job.error if job else None,
               "trace_url": job.trace_url if job else None}
        if status == "awaiting_review":
            out["review"] = pipeline.pending_review(state)
        if status == "done":
            v = state.values
            out["approved"] = bool(v.get("review", {}).get("approve"))
            out["report"] = f"/runs/{run_id}/report" if v.get("report_path") else None
        return out

    def stream(self, run_id: str, timeout_s: float = 15.0):
        """Server-Sent Events: past events first, then new ones, until a terminal status."""
        job = self.jobs.get(run_id)
        if job is None:
            yield _sse("status", self.status(run_id))
            return
        sent = 0
        while True:
            with self.changed:
                if sent == len(job.events) and job.status not in TERMINAL:
                    self.changed.wait(timeout_s)
                new, status = job.events[sent:], job.status
            for e in new:
                yield _sse(e["kind"], e)
            sent += len(new)
            if status in TERMINAL and sent == len(job.events):
                return
            if not new:
                yield ": keep-alive\n\n"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def default_graph():
    """The pipeline graph on the CLI's SQLite checkpoints, shared across threads."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    pipeline.CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(pipeline.CHECKPOINT_DB), check_same_thread=False)
    return pipeline.build_graph(SqliteSaver(conn))


def create_app(graph=None) -> FastAPI:
    runner: Runner | None = Runner(graph) if graph is not None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal runner
        if runner is None:
            runner = Runner(default_graph())
        app.state.runner = runner
        yield
        runner.pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="SurgiDoc", lifespan=lifespan,
                  description="Research demo on public data. Not for clinical use.")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/runs", status_code=202)
    def create_run(video: Annotated[UploadFile | None, File()] = None,
                   source: Annotated[str | None, Form()] = None):
        if (video is None) == (source is None):
            raise HTTPException(422, "send exactly one of: a video file, or source")
        run_id = uuid.uuid4().hex[:12]
        if source is not None:
            if not ALLOW_LOCAL_SOURCE:
                raise HTTPException(403, "server-side sources are disabled")
            if not Path(source).is_dir():
                raise HTTPException(422, f"not a directory: {source}")
            job = runner.submit_run(run_id, None, Path(source))
            return {"id": run_id, "status": job.status}
        path = UPLOADS_DIR / run_id / ("video" + Path(video.filename or "").suffix.lower())
        path.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with path.open("wb") as f:
            while chunk := video.file.read(2**20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    f.close()
                    shutil.rmtree(path.parent)
                    raise HTTPException(413, f"upload larger than {MAX_UPLOAD_BYTES // 2**20} MB")
                f.write(chunk)
        label = Path(video.filename or "upload").name
        job = runner.submit_run(run_id, path, None, label=label)
        return {"id": run_id, "status": job.status}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return runner.status(run_id)

    @app.get("/runs/{run_id}/events")
    def events(run_id: str):
        runner.status(run_id)  # 404 for unknown runs
        return StreamingResponse(runner.stream(run_id), media_type="text/event-stream")

    @app.post("/runs/{run_id}/review", status_code=202)
    def review(run_id: str, body: Review) -> dict:
        if runner.status(run_id)["status"] != "awaiting_review":
            raise HTTPException(409, "run is not awaiting review")
        job = runner.submit_review(run_id, body)
        return {"id": run_id, "status": job.status}

    @app.get("/runs/{run_id}/report", response_class=PlainTextResponse)
    def report(run_id: str):
        s = runner.status(run_id)
        path = runner.checkpoint(run_id).values.get("report_path")
        if s["status"] != "done" or not path:
            raise HTTPException(404, "no report: the run is not approved and finished")
        return PlainTextResponse(Path(path).read_text(), media_type="text/markdown")

    return app


app = create_app()
