"""Client-side rate limiting: a 60-second sliding window of requests and tokens per model."""

import re
import time
from collections import deque
from collections.abc import Callable

WINDOW_S = 60.0
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(text: str) -> float | None:
    """Parse Groq reset headers such as '1.065s', '1m26.4s', '450ms' or a bare '7'."""
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    parts = _DURATION_RE.findall(text)
    if not parts:
        return None
    return sum(float(value) * _UNIT_S[unit] for value, unit in parts)


def estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    """Conservative token estimate: about 3.5 characters per token, plus the output budget."""
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return int(chars / 3.5) + max_tokens


class Throttle:
    def __init__(
        self,
        rpm: int,
        tpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpm, self.tpm = rpm, tpm
        self._clock, self._sleep = clock, sleep
        self._events: deque[tuple[float, int]] = deque()  # (time, estimated tokens)
        self._remote_block_until = 0.0

    def acquire(self, tokens: int) -> float:
        """Block until a request of `tokens` fits the window. Returns seconds waited."""
        waited = 0.0
        while True:
            now = self._clock()
            while self._events and now - self._events[0][0] >= WINDOW_S:
                self._events.popleft()
            used = sum(t for _, t in self._events)
            fits = len(self._events) < self.rpm and (used + tokens <= self.tpm or not self._events)
            if fits and now >= self._remote_block_until:
                self._events.append((now, tokens))
                return waited
            if now < self._remote_block_until:
                delay = self._remote_block_until - now
            else:
                delay = self._events[0][0] + WINDOW_S - now
            delay = max(delay, 0.01)
            self._sleep(delay)
            waited += delay

    def update(self, headers: dict) -> None:
        """Honour the server's view: if its token budget is spent, wait for its reset."""
        headers = {k.lower(): v for k, v in headers.items()}
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset = parse_duration(headers.get("x-ratelimit-reset-tokens", ""))
        if remaining is not None and reset is not None and float(remaining) <= 0:
            self._remote_block_until = max(self._remote_block_until, self._clock() + reset)
