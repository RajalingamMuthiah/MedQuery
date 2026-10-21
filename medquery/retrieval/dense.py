"""Dense retrieval: cosine similarity over a numpy matrix (about 1,100 chunks, so no ANN index)."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medquery import paths
from medquery.retrieval.embed import Embedder, cache_path, embed_cached, normalise

CHUNKS = paths.DATA_PROCESSED / "chunks.jsonl"


def load_chunks(path: Path = CHUNKS, include_quiz: bool = False) -> list[dict]:
    """Chunk records from the build. Quiz chunks are excluded unless explicitly requested."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m medquery.ingest build` first")
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [c for c in chunks if include_quiz or c["kind"] != "quiz"]


@dataclass(frozen=True)
class Hit:
    chunk: dict
    score: float
    rank: int  # 1-based


class DenseIndex:
    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        self.chunks = chunks
        self.vectors = normalise(vectors.astype(np.float32))

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        scores = self.vectors @ normalise(query_vector.astype(np.float32))
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i]), rank) for rank, i in enumerate(top, 1)]


class Retriever:
    """Query embedding plus dense search. `retrieve` also returns its latency in seconds."""

    def __init__(self, embedder: Embedder, chunks: list[dict], embed_cache: Path | None = None):
        self.embedder = embedder
        path = embed_cache or cache_path(embedder.model_name)
        vectors, self.newly_embedded = embed_cached([c["text"] for c in chunks], embedder, path)
        self.index = DenseIndex(chunks, vectors)

    def retrieve(self, question: str, k: int) -> tuple[list[Hit], float]:
        started = time.perf_counter()
        hits = self.index.search(self.embedder.embed_query(question), k)
        return hits, time.perf_counter() - started
