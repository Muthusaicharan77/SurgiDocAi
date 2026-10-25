"""Guideline search: dense retrieval in Qdrant, then cross-encoder reranking.

Two stages because they trade off differently. The bi-encoder embeds query and
passages separately, so all passages are embedded once and search is a fast
vector lookup, but it only compares two summaries. The cross-encoder reads query
and passage together and judges relevance much better, but must run once per
pair, so it only rescores the top candidates.

    python -m surgidoc.retrieval.index build
    python -m surgidoc.retrieval.index search "critical view of safety before clipping"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel

from .chunking import Passage, load_corpus

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Chosen by the retrieval eval (RESULTS.md): hit@3 0.89 vs 0.59 for bge-reranker-base,
# at about 25 s per query on CPU.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# bge embedding models expect this prefix on queries (not on passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION = "guidelines"
DEFAULT_PATH = Path("data/qdrant")


class Hit(BaseModel):
    passage: Passage
    dense_score: float
    rerank_score: float | None = None


class GuidelineIndex:
    def __init__(self, path: Path = DEFAULT_PATH, rerank: bool = True,
                 rerank_model: str = RERANK_MODEL) -> None:
        from qdrant_client import QdrantClient
        from sentence_transformers import CrossEncoder, SentenceTransformer

        # Local file mode; the same client talks to a Qdrant server given a URL.
        self.client = QdrantClient(path=str(path))
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.reranker = CrossEncoder(rerank_model, device="cpu") if rerank else None

    def build(self, passages: list[Passage]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        vectors = self.embedder.encode([p.text for p in passages], normalize_embeddings=True,
                                       batch_size=32, show_progress_bar=False)
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            COLLECTION, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE))
        self.client.upsert(COLLECTION, points=[
            PointStruct(id=i, vector=v.tolist(), payload=p.model_dump())
            for i, (p, v) in enumerate(zip(passages, vectors))
        ])

    def search(self, query: str, k: int = 5, candidates: int = 25) -> list[Hit]:
        vector = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        points = self.client.query_points(COLLECTION, query=vector.tolist(),
                                          limit=candidates if self.reranker else k).points
        hits = [Hit(passage=Passage(**p.payload), dense_score=p.score) for p in points]
        if self.reranker is None:
            return hits[:k]
        scores = self.reranker.predict([(query, h.passage.text) for h in hits])
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:k]


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--no-rerank", action="store_true")
    args = p.parse_args()

    if args.cmd == "build":
        passages = load_corpus()
        GuidelineIndex(rerank=False).build(passages)
        print(f"indexed {len(passages)} passages into {DEFAULT_PATH}")
    else:
        index = GuidelineIndex(rerank=not args.no_rerank)
        for h in index.search(args.query, args.k):
            score = h.rerank_score if h.rerank_score is not None else h.dense_score
            print(f"{score:7.3f}  {h.passage.id:20s} {h.passage.section[:60]}")
            print(f"         {h.passage.text[:160]}...")


if __name__ == "__main__":
    main()
