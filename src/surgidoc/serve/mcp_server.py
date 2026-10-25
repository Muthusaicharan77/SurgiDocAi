"""MCP server: SurgiDoc's vision and retrieval tools for any MCP client.

    python -m surgidoc.serve.mcp_server            # stdio, for Claude Desktop / Claude Code

Claude Code: `claude mcp add surgidoc -- /path/to/.venv/bin/python -m surgidoc.serve.mcp_server`
(run from the repo root, so weights/ and data/ resolve).

Tools take image paths on the machine the server runs on. Every result says which
classes have a measured test score, so a client can tell validated output from
unvalidated output, as the report does. Research demo, not for clinical use.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from PIL import Image
from pydantic import BaseModel
from scipy import ndimage

from surgidoc.agent.facts import MIN_VISIBLE_AREA, validated_classes
from surgidoc.vision.classes import CLASS_NAMES
from surgidoc.vision.phase import PHASES

server = MCPServer(
    name="surgidoc",
    instructions=(
        "Tools for laparoscopic cholecystectomy frames: segment anatomy and instruments, "
        "estimate the surgical phase, and search open-access guidelines. Segmentation "
        "classes without measured accuracy and all phase estimates are unvalidated. "
        "Research demo on public data, not for clinical use."),
)

MAX_IMAGES = 64


class Structure(BaseModel):
    area: float  # share of the frame, 0-1
    components: int  # connected regions
    visible: bool  # area >= 0.5% of the frame, the threshold the report uses
    validated: bool  # the class has a measured test score


class Segmentation(BaseModel):
    image_path: str
    structures: dict[str, Structure]


class PhaseFrame(BaseModel):
    image_path: str
    phase: str
    confidence: float


class Phases(BaseModel):
    phases: list[str]
    validated: bool  # always False: no measured accuracy on this data
    frames: list[PhaseFrame]


class GuidelinePassage(BaseModel):
    id: str
    document: str
    section: str
    text: str
    url: str
    license: str
    score: float


class Guidelines(BaseModel):
    query: str
    passages: list[GuidelinePassage]


@cache
def perceiver():
    from surgidoc.agent.graph import PHASE_WEIGHTS, SEG_WEIGHTS
    from surgidoc.agent.perception import Perceiver

    return Perceiver(SEG_WEIGHTS, PHASE_WEIGHTS)


@cache
def guideline_index():
    import atexit

    from surgidoc.retrieval.index import GuidelineIndex

    index = GuidelineIndex()
    atexit.register(index.client.close)  # close Qdrant before interpreter teardown
    return index


def _open(paths: list[str]) -> list[Image.Image]:
    if not paths:
        raise ValueError("no image paths given")
    if len(paths) > MAX_IMAGES:
        raise ValueError(f"at most {MAX_IMAGES} images per call")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise ValueError(f"not found: {missing}")
    return [Image.open(p).convert("RGB") for p in paths]


@server.tool(structured_output=True)
def segment_frame(image_path: str) -> Segmentation:
    """Segment one endoscopic frame into 13 CholecSeg8k classes (U-Net).

    Returns each detected class with its share of the frame, its number of connected
    regions, whether it counts as visible (at least 0.5% of the frame), and whether the
    class has a measured test score. Unvalidated detections (e.g. cystic duct, blood)
    have no measured accuracy.
    """
    mask = perceiver().segment(_open([image_path]))[0]
    validated = validated_classes()
    structures = {}
    for c in sorted(set(mask.flatten().tolist())):
        name = CLASS_NAMES[c]
        region = mask == c
        area = float(region.mean())
        structures[name] = Structure(area=round(area, 4),
                                     components=int(ndimage.label(region)[1]),
                                     visible=area >= MIN_VISIBLE_AREA,
                                     validated=name in validated)
    return Segmentation(image_path=image_path, structures=structures)


@server.tool(structured_output=True)
def classify_phase(image_paths: list[str]) -> Phases:
    """Estimate the Cholec80 surgical phase of each frame (Trans-SVNet ResNet50, unsmoothed).

    The phase model has no measured accuracy on this project's data; treat results as
    estimates. Pass frames in video order; per-frame predictions flicker, so a client
    should smooth over neighbouring frames before drawing conclusions.
    """
    probs = perceiver().phase.predict_proba(_open(image_paths))
    return Phases(phases=PHASES, validated=False, frames=[
        PhaseFrame(image_path=p, phase=PHASES[int(row.argmax())],
                   confidence=round(float(row.max()), 4))
        for p, row in zip(image_paths, probs)])


@server.tool(structured_output=True)
def retrieve_guideline(query: str, k: int = 3) -> Guidelines:
    """Search open-access (CC BY) cholecystectomy guidelines.

    Dense retrieval over ~220 passages, reranked by a cross-encoder. Each passage has
    an id, source document, section, licence and URL, for citation.
    """
    k = max(1, min(k, 10))
    hits = guideline_index().search(query, k=k)
    return Guidelines(query=query, passages=[
        GuidelinePassage(
            id=h.passage.id, document=h.passage.doc_title, section=h.passage.section,
            text=h.passage.text, url=h.passage.url, license=h.passage.license,
            score=round(h.rerank_score if h.rerank_score is not None else h.dense_score, 4))
        for h in hits])


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
