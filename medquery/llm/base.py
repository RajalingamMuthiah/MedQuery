"""Provider-neutral types for chat completions."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    finish_reason: str
    usage: dict = field(default_factory=dict)  # prompt/completion/reasoning tokens
    latency_s: float = 0.0  # the original API call's latency, also for cache hits
    cached: bool = False
    wait_s: float = 0.0  # time spent in the client-side throttle (0 for cache hits)


class LLMError(Exception):
    """A request failed and will not be retried."""


class QuotaExhausted(LLMError):
    """A daily limit was hit. Stop now and resume later."""

    def __init__(self, message: str, retry_after_s: float | None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class TransportError(Exception):
    """An HTTP-level failure. status is None for timeouts and connection errors."""

    def __init__(self, status: int | None, headers: dict, message: str):
        super().__init__(message)
        self.status = status
        self.headers = {k.lower(): v for k, v in headers.items()}


class Transport(Protocol):
    """Sends one chat-completion request; returns (response JSON, response headers)."""

    def send(self, request: dict) -> tuple[dict, dict]: ...


class LLM(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 600,
        seed: int | None = 0,
        reasoning_effort: str | None = None,
        response_format: dict | None = None,
    ) -> Completion: ...
