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


def trusted_networks(spec: str) -> list:
    """Parse MEDQUERY_TRUSTED_PROXIES: comma-separated IPs or CIDR ranges ('' = none)."""
    import ipaddress

    networks = []
    for part in (p.strip() for p in spec.split(",")):
        if part:
            networks.append(ipaddress.ip_network(part, strict=False))
    return networks


def client_ip(peer: str, forwarded_for: str | None, trusted: list) -> str:
    """The address to rate-limit. X-Forwarded-For is believed only when the direct peer is a
    trusted proxy; then the client is the rightmost hop that is not itself a trusted proxy.
    From anyone else the header is ignored, so spoofing it changes nothing."""
    import ipaddress

    def is_trusted(address: str) -> bool:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        return any(ip in network for network in trusted)

    if not trusted or not forwarded_for or not is_trusted(peer):
        return peer
    for hop in reversed([h.strip() for h in forwarded_for.split(",") if h.strip()]):
        try:
            ipaddress.ip_address(hop)
        except ValueError:
            return peer  # a malformed chain is not trusted
        if not is_trusted(hop):
            return hop
    return peer  # every hop is a trusted proxy
