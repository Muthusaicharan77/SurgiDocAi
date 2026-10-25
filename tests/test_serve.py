import numpy as np
import pytest

from surgidoc.agent.perception import list_frames


def make_video(path, seconds=2.0, fps=30, size=(64, 48)):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(int(seconds * fps)):
            img = np.full((size[1], size[0], 3), i * 4 % 255, np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
    return path


def test_extract_frames_samples_by_time_and_names_by_25fps_index(tmp_path):
    from surgidoc.serve.frames import extract_frames

    video = make_video(tmp_path / "clip.mp4", seconds=2.0, fps=30)
    n = extract_frames(video, tmp_path / "frames", sample_fps=5)
    refs = list_frames(tmp_path / "frames")
    assert n == len(refs) == 10
    assert [r.frame_number for r in refs] == list(range(0, 50, 5))
    assert refs[-1].time_s == pytest.approx(1.8)


def test_extract_frames_rejects_overlong_video(tmp_path):
    from surgidoc.serve.frames import extract_frames

    video = make_video(tmp_path / "clip.mp4", seconds=2.0, fps=30)
    with pytest.raises(ValueError, match="longer than"):
        extract_frames(video, tmp_path / "frames", sample_fps=5, max_frames=3)


class FakeGraph:
    """The interface the API uses from the compiled LangGraph graph, without models."""

    NODES = ("sample_frames", "perceive", "build_timeline", "select_key_frames", "retrieve",
             "draft_report", "verify", "judge", "drop_unsupported")

    def __init__(self, tmp_path, fail=False):
        self.tmp, self.fail, self.threads = tmp_path, fail, {}

    def _thread(self, config):
        return config["configurable"]["thread_id"]

    def stream(self, inputs, config, stream_mode):
        from surgidoc.agent.perception import list_frames

        if self.fail:
            raise RuntimeError("perception failed")
        frames = list_frames(inputs["source"])
        for node in self.NODES:
            yield {node: {}}
        claim = {"id": "c1", "claim_type": "phase", "text": "Preparation", "basis": "none"}
        self.threads[self._thread(config)] = {
            "values": {**inputs, "frames": len(frames), "drafts": 1},
            "interrupt": {"claims": [claim], "dropped": []}}
        yield {"__interrupt__": ()}

    def get_state(self, config):
        from types import SimpleNamespace as NS

        t = self.threads.get(self._thread(config), {"values": {}, "interrupt": None})
        tasks = [NS(interrupts=[NS(value=t["interrupt"])])] if t["interrupt"] else []
        return NS(values=t["values"], tasks=tasks, next=("human_review",) if tasks else ())

    def invoke(self, command, config):
        t = self.threads[self._thread(config)]
        report = self.tmp / f"{self._thread(config)}.md"
        report.write_text(f"# Report\n\nnotes: {command.resume['notes']}\n")
        t["values"] |= {"review": command.resume, "report_path": str(report)}
        t["interrupt"] = None
        return t["values"]


def wait_for(client, run_id, statuses, timeout=10):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} still {body['status']}")


@pytest.fixture
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from surgidoc.serve import api as api_mod

    monkeypatch.setattr(api_mod, "UPLOADS_DIR", tmp_path / "uploads")
    graph = FakeGraph(tmp_path)

    def client(**kwargs):
        if kwargs:
            graph.__dict__.update(kwargs)
        return TestClient(api_mod.create_app(graph))
    return client, tmp_path


def test_upload_run_review_report(api):
    make_client, tmp_path = api
    video = make_video(tmp_path / "clip.mp4", seconds=1.0, fps=25)
    with make_client() as client:
        r = client.post("/runs", files={"video": ("clip.mp4", video.read_bytes(), "video/mp4")})
        assert r.status_code == 202
        run_id = r.json()["id"]

        body = wait_for(client, run_id, {"awaiting_review", "failed"})
        assert body["status"] == "awaiting_review", body
        assert body["nodes_done"][0] == "extract_frames" and "judge" in body["nodes_done"]
        assert body["review"]["claims"][0]["id"] == "c1"
        values = client.app.state.runner.checkpoint(run_id).values
        assert values["label"] == "clip.mp4" and values["every_n"] == 1  # every extracted frame
        assert client.get(f"/runs/{run_id}/report").status_code == 404  # not reviewed yet

        events = client.get(f"/runs/{run_id}/events").text
        assert "event: node" in events and '"status": "awaiting_review"' in events

        r = client.post(f"/runs/{run_id}/review", json={"approve": True, "notes": "looks right"})
        assert r.status_code == 202
        body = wait_for(client, run_id, {"done", "failed"})
        assert body["status"] == "done" and body["approved"], body
        report = client.get(f"/runs/{run_id}/report")
        assert report.status_code == 200 and "notes: looks right" in report.text
        # A second review is refused: the run is no longer waiting.
        assert client.post(f"/runs/{run_id}/review", json={"approve": True}).status_code == 409


def test_failures_and_bad_requests_are_reported(api):
    make_client, tmp_path = api
    video = make_video(tmp_path / "clip.mp4", seconds=0.5, fps=25)
    with make_client(fail=True) as client:
        run_id = client.post("/runs", files={"video": ("c.mp4", video.read_bytes())}).json()["id"]
        body = wait_for(client, run_id, {"failed"})
        assert "perception failed" in body["error"]

        assert client.get("/runs/nope").status_code == 404
        assert client.post("/runs").status_code == 422  # neither video nor source
        assert client.post("/runs", data={"source": str(tmp_path)}).status_code == 403  # off
        not_video = client.post("/runs", files={"video": ("x.mp4", b"not a video")}).json()["id"]
        assert wait_for(client, not_video, {"failed"})["error"]


def test_mcp_tools_over_an_in_memory_client(tmp_path, monkeypatch):
    import anyio
    from mcp.client.client import Client
    from PIL import Image

    from surgidoc.retrieval.chunking import Passage
    from surgidoc.retrieval.index import Hit
    from surgidoc.serve import mcp_server as m

    class FakePerceiver:
        class phase:
            @staticmethod
            def predict_proba(images):
                p = np.zeros((len(images), 7))
                p[:, 1] = 0.9
                p[:, 0] = 0.1
                return p

        def segment(self, images):
            mask = np.full((20, 20), 2, np.int64)  # liver
            mask[:4] = 5  # grasper, 20% of the frame
            mask[19, 19] = 8  # one pixel (0.25%) of cystic duct: detected, not visible
            return [mask for _ in images]

    class FakeIndex:
        def search(self, query, k):
            p = Passage(id="P1", doc_key="D", doc_title="Doc", section="S", text="CVS first.",
                        url="https://example.org", license="cc by")
            return [Hit(passage=p, dense_score=0.5, rerank_score=0.9)][:k]

    monkeypatch.setattr(m, "perceiver", lambda: FakePerceiver())
    monkeypatch.setattr(m, "guideline_index", lambda: FakeIndex())
    image = tmp_path / "f.png"
    Image.new("RGB", (8, 8)).save(image)

    async def calls():
        async with Client(m.server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            seg = await client.call_tool("segment_frame", {"image_path": str(image)})
            phase = await client.call_tool("classify_phase", {"image_paths": [str(image)] * 2})
            guide = await client.call_tool("retrieve_guideline", {"query": "cvs", "k": 1})
            missing = await client.call_tool("segment_frame", {"image_path": "/nope.png"})
            return names, seg, phase, guide, missing

    names, seg, phase, guide, missing = anyio.run(calls)
    assert names == {"segment_frame", "classify_phase", "retrieve_guideline"}
    s = seg.structured_content["structures"]
    assert s["liver"]["visible"] and s["liver"]["validated"] and s["grasper"]["area"] == 0.2
    assert not s["cystic_duct"]["visible"] and not s["cystic_duct"]["validated"]
    frames = phase.structured_content["frames"]
    assert [f["phase"] for f in frames] == ["calot_triangle_dissection"] * 2
    assert phase.structured_content["validated"] is False
    assert guide.structured_content["passages"][0]["id"] == "P1"
    assert missing.is_error
