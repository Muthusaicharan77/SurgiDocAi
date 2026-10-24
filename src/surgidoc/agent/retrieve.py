"""Node retrieve: build search queries from the timeline and fetch guideline passages."""

from __future__ import annotations

from functools import cache

from surgidoc.retrieval.chunking import Passage

from .facts import Timeline

# Asked for every video: what a cholecystectomy report should document, and the
# safety step the vision models cannot verify.
STANDING_QUERIES = [
    "what should the operative note of a laparoscopic cholecystectomy document",
    "critical view of safety must be achieved before clipping and dividing structures",
]
PHASE_QUERIES = {
    "preparation": "port placement and exposure of the gallbladder at the start of "
                   "laparoscopic cholecystectomy",
    "calot_triangle_dissection": "dissection of the hepatocystic triangle to identify the "
                                 "cystic duct and cystic artery",
    "clipping_cutting": "clipping and dividing the cystic duct and cystic artery",
    "gallbladder_dissection": "separating the gallbladder from the liver bed",
    "gallbladder_packaging": "retrieval of the gallbladder in an extraction bag",
    "cleaning_coagulation": "haemostasis and irrigation of the liver bed after cholecystectomy",
    "gallbladder_retraction": "extraction of the gallbladder and port closure",
}
EVENT_QUERIES = {
    "possible_bleeding": "management of intraoperative bleeding during laparoscopic "
                         "cholecystectomy",
}
PER_QUERY = 3


@cache
def _index():
    import atexit

    from surgidoc.retrieval.index import GuidelineIndex

    index = GuidelineIndex()
    atexit.register(index.client.close)  # close Qdrant before interpreter teardown
    return index


def queries_for(timeline: Timeline) -> list[str]:
    queries = list(STANDING_QUERIES)
    for phase in dict.fromkeys(s.phase for s in timeline.segments):
        queries.append(PHASE_QUERIES[phase])
    for kind in dict.fromkeys(e.kind for e in timeline.events):
        if kind in EVENT_QUERIES:
            queries.append(EVENT_QUERIES[kind])
    return queries


def retrieve(timeline: Timeline) -> tuple[list[Passage], list[str]]:
    """Top passages for each query, deduplicated, in first-seen order."""
    queries = queries_for(timeline)
    passages: dict[str, Passage] = {}
    for q in queries:
        for hit in _index().search(q, k=PER_QUERY):
            passages.setdefault(hit.passage.id, hit.passage)
    return list(passages.values()), queries
