"""Split JATS full-text articles into citable passages.

A passage is a few consecutive paragraphs from one section, about 120-300 words:
long enough to carry one recommendation with its context, short enough that a
citation points at something specific. Each passage keeps its section path
("Section 3. Surgical treatment of ACC > ...") so the report can say where it is.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel

MIN_WORDS = 120
MAX_WORDS = 300

# Back matter and review methodology: no clinical content. Broad "methods" is not
# skipped, because WSES-BDI keeps its clinical evidence under "Guideline scope and
# methods > Literature review".
SKIP_SECTIONS = re.compile(
    r"supplementary|author|contribution|institutional review|informed consent|data availability|"
    r"conflict|competing interest|funding|acknowledg|abbreviation|declaration|ethics|"
    r"search strategy|study selection|data extraction|statistical analysis|quality assessment|"
    r"eligibility|risk of bias",
    re.IGNORECASE,
)
_CONTAINERS = {"boxed-text", "list", "list-item", "disp-quote", "def-list", "def-item", "def"}


class Passage(BaseModel):
    id: str  # e.g. "WSES-ACC-2020-042", what report claims cite
    doc_key: str
    doc_title: str
    section: str
    text: str
    url: str
    license: str


def _text(el: ET.Element) -> str:
    return " ".join("".join(el.itertext()).split())


def _paragraphs(sec: ET.Element, path: list[str]):
    """(section path, paragraph text) in document order, recursing into subsections."""
    title = sec.find("title")
    if title is not None:
        path = [*path, _text(title)]
    if path and SKIP_SECTIONS.search(path[-1]):
        return
    for child in sec:
        if child.tag == "sec":
            yield from _paragraphs(child, path)
        elif child.tag == "p":
            yield " > ".join(path), _text(child)
        elif child.tag in _CONTAINERS:
            for p in child.iter("p"):
                yield " > ".join(path), _text(p)


def _split_long(text: str) -> list[str]:
    """Split a paragraph longer than MAX_WORDS at sentence ends."""
    if len(text.split()) <= MAX_WORDS:
        return [text]
    parts, current = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(" ".join([*current, sentence]).split()) > MAX_WORDS:
            parts.append(" ".join(current))
            current = []
        current.append(sentence)
    parts.append(" ".join(current))
    return parts


def chunk_article(xml_path: Path, meta: dict) -> list[Passage]:
    root = ET.parse(xml_path).getroot()
    units = []
    abstract = root.find(".//abstract")
    if abstract is not None:
        units += [("Abstract", _text(p)) for p in abstract.iter("p")]
    body = root.find(".//body")
    if body is not None:
        for sec in body.findall("sec"):
            units += list(_paragraphs(sec, []))

    passages, buffer, buffer_section = [], [], None

    def flush():
        if buffer:
            passages.append(" ".join(buffer))
            sections.append(buffer_section)
            buffer.clear()

    sections: list[str] = []
    for section, para in units:
        if not para:
            continue
        for piece in _split_long(para):
            if section != buffer_section or len(" ".join([*buffer, piece]).split()) > MAX_WORDS:
                flush()
                buffer_section = section
            buffer.append(piece)
            if len(" ".join(buffer).split()) >= MIN_WORDS:
                flush()
    flush()

    return [
        Passage(id=f"{meta['key']}-{i + 1:03d}", doc_key=meta["key"], doc_title=meta["title"],
                section=section, text=text, url=meta["url"], license=meta["license"])
        for i, (section, text) in enumerate(zip(sections, passages))
    ]


def load_corpus(folder: Path = Path("data/guidelines")) -> list[Passage]:
    passages = []
    for meta_path in sorted(folder.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        passages += chunk_article(meta_path.with_suffix(".xml"), meta)
    return passages
