"""Configurable retrieval: dense, optional BM25 with RRF fusion, optional reranking, optional
contextual chunk headers. Headers change only the text that is embedded, BM25-indexed and
reranked; the generator always sees the original chunk text."""

import time
from pathlib import Path

from medquery.retrieval.bm25 import BM25
from medquery.retrieval.dense import DenseIndex, Hit
from medquery.retrieval.embed import Embedder, cache_path, embed_cached
from medquery.retrieval.fusion import RRF_K, rrf
from medquery.retrieval.rerank import CrossEncoder, rerank

FUSE_DEPTH = 50  # candidates taken from each list before fusion


def index_text(chunk: dict, headers: bool) -> str:
    if not headers:
        return chunk["text"]
    context = " | ".join(p for p in (chunk.get("title", ""), chunk.get("section", "")) if p)
    return f"{context}\n{chunk['text']}" if context else chunk["text"]


class HybridRetriever:
    """Same interface as dense.Retriever: retrieve(question, k) -> (hits, seconds)."""

    def __init__(
        self,
        embedder: Embedder,
        chunks: list[dict],
        *,
        use_bm25: bool,
        headers: bool = False,
        reranker: CrossEncoder | None = None,
        rerank_n: int = 20,
        fuse_k: int = RRF_K,
        fuse_depth: int = FUSE_DEPTH,
        embed_cache: Path | None = None,
    ):
        self.embedder = embedder
        self.chunks = chunks
        self.headers = headers
        self.texts = {c["id"]: index_text(c, headers) for c in chunks}
        path = embed_cache or cache_path(embedder.model_name)
        vectors, self.newly_embedded = embed_cached(
            [self.texts[c["id"]] for c in chunks], embedder, path
        )
        self.dense = DenseIndex(chunks, vectors)
        self.bm25 = BM25([self.texts[c["id"]] for c in chunks]) if use_bm25 else None
        self.reranker, self.rerank_n = reranker, rerank_n
        self.fuse_k, self.fuse_depth = fuse_k, fuse_depth
        self._by_id = {c["id"]: c for c in chunks}

    def retrieve(self, question: str, k: int) -> tuple[list[Hit], float]:
        started = time.perf_counter()
        depth = max(self.fuse_depth, k, self.rerank_n if self.reranker else 0)
        dense = self.dense.search(self.embedder.embed_query(question), depth)
        if self.bm25 is not None:
            lexical = [self.chunks[i]["id"] for i, _ in self.bm25.top(question, depth)]
            fused = rrf([[h.chunk["id"] for h in dense], lexical], self.fuse_k)
            ranked = [(self._by_id[cid], score) for cid, score in fused]
        else:
            ranked = [(h.chunk, h.score) for h in dense]
        self.last_stats: dict = {}
        if self.reranker is not None:
            candidates = [c for c, _ in ranked]
            ranked = rerank(
                question, candidates, lambda c: self.texts[c["id"]], self.reranker, self.rerank_n
            )
            elapsed = time.perf_counter() - started  # the length check is not part of latency
            head = [self.texts[c["id"]] for c in candidates[: self.rerank_n]]
            self.last_stats = {
                "rerank_pairs": len(head),
                "rerank_truncated": self.reranker.count_truncated(question, head),
            }
            hits = [Hit(c, score, rank) for rank, (c, score) in enumerate(ranked[:k], 1)]
            return hits, elapsed
        hits = [Hit(chunk, score, rank) for rank, (chunk, score) in enumerate(ranked[:k], 1)]
        return hits, time.perf_counter() - started
