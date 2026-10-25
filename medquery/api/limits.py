"""Request limits that protect the Groq free tier: a per-IP rate limit (in memory) and a daily
cap on Groq-calling requests shared by every user (persisted, so a restart does not reset it)."""

import json
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from medquery.fileio import atomic_write


class RateLimiter:
    """At most `per_minute` requests per client in any 60-second window."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute, self.clock = per_minute, clock
        self._seen: dict[str, deque] = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now, window = self.clock(), self._seen[client]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


def utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


class DailyCap:
    """A shared daily count of Groq-calling requests, reset at 00:00 UTC. Only requests that
    actually reached Groq are counted (cached repeats are free)."""

    def __init__(self, path: Path, cap: int, today: Callable[[], str] = utc_today):
        self.path, self.cap, self.today = path, cap, today

    def _count(self) -> int:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return 0
        return data["count"] if data.get("date") == self.today() else 0

    def remaining(self) -> int:
        return max(0, self.cap - self._count())

    def record(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_write(self.path) as f:
            json.dump({"date": self.today(), "count": self._count() + 1}, f)
