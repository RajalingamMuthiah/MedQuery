"""Cross-encoder reranking (fastembed ONNX on CPU, no torch)."""

import os
from pathlib import Path
from typing import Protocol

from medquery import paths

MODEL_CACHE = paths.ROOT / "data" / "cache" / "fastembed"
RERANKERS = {  # model id -> licence (recorded in reports)
    "BAAI/bge-reranker-base": "MIT",
}


class CrossEncoder(Protocol):
    model_name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...

    def count_truncated(self, query: str, texts: list[str]) -> int:
        """How many (query, text) pairs exceed the model's max input length."""
        ...


class FastEmbedReranker:
    def __init__(self, model_name: str, cache_dir: Path = MODEL_CACHE):
        if model_name not in RERANKERS:
            raise ValueError(f"unknown reranker {model_name!r}; add it with its licence first")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # loads ONNX Runtime

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name, cache_dir=str(cache_dir))

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]

    def _tokenizer(self):
        """The model's own tokenizer, with truncation turned off so lengths are true lengths."""
        if not hasattr(self, "_counting_tokenizer"):
            from tokenizers import Tokenizer

            source = self._model.model.tokenizer  # fastembed's loaded Hugging Face tokenizer
            tokenizer = Tokenizer.from_str(source.to_str())
            tokenizer.no_truncation()
            tokenizer.no_padding()
            self._counting_tokenizer = tokenizer
            self.max_length = source.truncation["max_length"] if source.truncation else 512
        return self._counting_tokenizer

    def count_truncated(self, query: str, texts: list[str]) -> int:
        tokenizer = self._tokenizer()
        return sum(len(tokenizer.encode(query, text).ids) > self.max_length for text in texts)


def rerank(
    query: str, candidates: list[dict], text_of, encoder: CrossEncoder, n: int
) -> list[tuple[dict, float]]:
    """Score the first n candidates with the cross-encoder and sort them by that score
    (stable, so equal scores keep the incoming order). Candidates beyond n are dropped."""
    head = candidates[:n]
    scores = encoder.score(query, [text_of(c) for c in head])
    order = sorted(range(len(head)), key=lambda i: -scores[i])
    return [(head[i], scores[i]) for i in order]
