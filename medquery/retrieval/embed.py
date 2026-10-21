"""Text embeddings (fastembed / ONNX, no torch) with an on-disk cache keyed by text hash."""

import hashlib
import io
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from medquery import config, paths
from medquery.fileio import atomic_write_bytes

EMBED_CACHE = paths.ROOT / "data" / "cache" / "embeddings"
MODEL_CACHE = paths.ROOT / "data" / "cache" / "fastembed"


class Embedder(Protocol):
    model_name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """BGE via fastembed; passage_embed/query_embed apply BGE's query instruction prefix."""

    def __init__(self, model_name: str = config.EMBED_MODEL, cache_dir: Path = MODEL_CACHE):
        # Windows without Developer Mode has no symlinks; the HF cache still works, so the
        # warning is noise in CLI output.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding  # imported lazily: it loads ONNX Runtime

        self.model_name = model_name
        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.query_embed([text]))), dtype=np.float32)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def cache_path(model_name: str, root: Path = EMBED_CACHE) -> Path:
    return root / f"{model_name.replace('/', '__')}.npz"


def embed_cached(texts: list[str], embedder: Embedder, path: Path) -> tuple[np.ndarray, int]:
    """Embed `texts`, reusing cached vectors by text hash. Returns (vectors, number computed)."""
    cached: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path) as data:
            cached = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))
    keys = [text_key(t) for t in texts]
    missing = sorted({k for k in keys if k not in cached})
    if missing:
        by_key = {text_key(t): t for t in texts}
        fresh = normalise(embedder.embed_passages([by_key[k] for k in missing]))
        cached.update(zip(missing, fresh, strict=True))
        buffer = io.BytesIO()
        all_keys = sorted(cached)
        np.savez(buffer, keys=np.array(all_keys), vectors=np.stack([cached[k] for k in all_keys]))
        atomic_write_bytes(path, buffer.getvalue())
    return np.stack([cached[k] for k in keys]), len(missing)
