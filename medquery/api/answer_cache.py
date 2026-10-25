"""Stored /ask responses for repeated questions: a bounded LRU keyed by the normalised question
and the live config identity, persisted in data/ so it survives restarts."""

import json
import re
import threading
import unicodedata
from collections import OrderedDict
from pathlib import Path

from medquery.cache.disk import cache_key
from medquery.fileio import atomic_write

MAX_ENTRIES = 500


def normalise(question: str) -> str:
    """Case, spacing and trailing punctuation do not make a different question."""
    text = unicodedata.normalize("NFKC", question).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[\s?.!]+$", "", text)


class AnswerCache:
    """path=None keeps it in memory only; enabled=False makes every lookup a miss."""

    def __init__(self, path: Path | None, max_entries: int = MAX_ENTRIES, enabled: bool = True):
        self.path, self.max_entries, self.enabled = path, max_entries, enabled
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        if enabled and path is not None and path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                self._entries = OrderedDict(stored.get("entries", []))
            except (json.JSONDecodeError, ValueError):
                self._entries = OrderedDict()  # a damaged cache is just empty

    @staticmethod
    def key(question: str, identity: str) -> str:
        return cache_key({"q": normalise(question), "config": identity})

    def get(self, question: str, identity: str) -> dict | None:
        if not self.enabled:
            return None
        with self._lock:
            k = self.key(question, identity)
            if k not in self._entries:
                return None
            self._entries.move_to_end(k)
            return self._entries[k]

    def put(self, question: str, identity: str, response: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            k = self.key(question, identity)
            self._entries[k] = response
            self._entries.move_to_end(k)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with atomic_write(self.path) as f:
                    json.dump({"entries": list(self._entries.items())}, f, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self._entries)
