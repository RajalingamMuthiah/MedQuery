"""A content-addressed JSON cache on disk (one file per key, written atomically)."""

import hashlib
import json
from pathlib import Path

from medquery.fileio import atomic_write


def cache_key(payload: dict) -> str:
    """Stable sha256 of a JSON-serialisable payload (key order does not matter)."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class JsonCache:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict) -> None:
        with atomic_write(self._path(key)) as f:
            json.dump(value, f, ensure_ascii=False)
