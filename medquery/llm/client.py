"""LLM client: disk cache, then throttle, then retries, then the provider transport."""

import random
import time
from collections.abc import Callable

from medquery.cache.disk import JsonCache, cache_key
from medquery.llm.base import Completion, LLMError, QuotaExhausted, Transport, TransportError
from medquery.llm.throttle import Throttle, estimate_tokens, parse_duration

MAX_TRIES = 6
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0
DAILY_LIMIT_RETRY_S = 120.0  # a retry-after longer than this means a daily quota, not a burst


class LLMClient:
    def __init__(
        self,
        transport: Transport,
        *,
        provider: str,
        cache: JsonCache | None,
        throttle_factory: Callable[[], Throttle],
        secrets: tuple[str, ...] = (),
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        rng: random.Random | None = None,
    ):
        self._transport = transport
        self._provider = provider
        self._cache = cache
        self._throttle_factory = throttle_factory
        self._throttles: dict[str, Throttle] = {}
        self._secrets = tuple(s for s in secrets if s)
        self._sleep, self._clock = sleep, clock
        self._rng = rng or random.Random(0)

    def __repr__(self) -> str:  # never include the transport (which holds the key)
        return f"LLMClient(provider={self._provider!r})"

    @staticmethod
    def _request(messages, model, temperature, max_tokens, seed, reasoning_effort, response_format):
        request = {"model": model, "messages": messages}
        if temperature is not None:  # None: the model only accepts its default temperature
            request["temperature"] = temperature
        request["max_completion_tokens"] = max_tokens
        if seed is not None:
            request["seed"] = seed
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            request["response_format"] = response_format
        return request

    def _throttle(self, model: str) -> Throttle:
        return self._throttles.setdefault(model, self._throttle_factory())

    def estimate_wait(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int = 600,
        expected_output_tokens: int | None = None,
        **_ignored,
    ) -> float:
        """Seconds the throttle would make this request wait now (nothing is booked)."""
        tokens = estimate_tokens(messages, max_tokens, expected_output_tokens)
        return self._throttle(model).estimate_wait(tokens)

    def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float | None = 0.0,
        max_tokens: int = 600,
        seed: int | None = 0,
        reasoning_effort: str | None = None,
        response_format: dict | None = None,
        expected_output_tokens: int | None = None,
    ) -> Completion:
        """expected_output_tokens: book this much output in the throttle instead of max_tokens
        (the live app; None keeps the conservative booking). It is not part of the request."""
        request = self._request(
            messages, model, temperature, max_tokens, seed, reasoning_effort, response_format
        )
        key = cache_key({"provider": self._provider, **request})
        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            return Completion(**hit, cached=True)

        throttle = self._throttle(model)
        wait_s = throttle.acquire(estimate_tokens(messages, max_tokens, expected_output_tokens))
        data, headers, latency_s = self._send_with_retry(request)
        throttle.update(headers)

        choice = data["choices"][0]
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        record = {
            "text": choice["message"].get("content") or "",
            "model": data.get("model", model),
            "finish_reason": choice.get("finish_reason") or "",
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": details.get("reasoning_tokens", 0),
            },
            "latency_s": round(latency_s, 4),
        }
        if self._cache is not None:
            self._cache.put(key, record)
        return Completion(**record, wait_s=round(wait_s, 3))

    def stream(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float | None = 0.0,
        max_tokens: int = 600,
        seed: int | None = 0,
        reasoning_effort: str | None = None,
        response_format: dict | None = None,
        expected_output_tokens: int | None = None,
    ) -> "CompletionStream":
        """Like complete(), but yields the answer text as it arrives. The same request is the
        cache key, so a streamed answer is stored exactly as complete() would store it, and a
        cached one is yielded in one piece. Iterate it; then .completion holds the result."""
        request = self._request(
            messages, model, temperature, max_tokens, seed, reasoning_effort, response_format
        )
        return CompletionStream(self, request, expected_output_tokens)

    def _send_with_retry(self, request: dict, send=None) -> tuple[dict, dict, float]:
        """Send with retries; `send` (default: the transport's send) may return a stream, in
        which case only failures before the first byte are retried."""
        send = send or self._transport.send
        last_error = ""
        for attempt in range(MAX_TRIES):
            started = self._clock()
            try:
                data, headers = send(request)
                return data, headers, self._clock() - started
            except TransportError as exc:
                message = self._redact(str(exc))
                last_error = f"HTTP {exc.status}: {message}" if exc.status else message
                retry_after = parse_duration(exc.headers.get("retry-after", ""))
                if exc.status == 429:
                    if "per day" in message.lower() or (retry_after or 0) > DAILY_LIMIT_RETRY_S:
                        raise QuotaExhausted(
                            f"daily rate limit reached for {request['model']}; resume later "
                            f"(retry-after {retry_after}s). {last_error}",
                            retry_after,
                        ) from None
                    delay = retry_after if retry_after is not None else self._backoff(attempt)
                elif exc.status is None or exc.status >= 500:
                    delay = self._backoff(attempt)
                else:
                    raise LLMError(last_error) from None
            if attempt + 1 < MAX_TRIES:
                self._sleep(delay)
        raise LLMError(f"gave up after {MAX_TRIES} tries: {last_error}")

    def _backoff(self, attempt: int) -> float:
        delay = min(BACKOFF_BASE_S * 2**attempt, BACKOFF_CAP_S)
        return delay + self._rng.uniform(0, delay / 4)

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text


class CompletionStream:
    """Text deltas of one streamed completion; .completion is set once iteration finishes."""

    def __init__(self, client: LLMClient, request: dict, expected_output_tokens: int | None):
        self._client, self._request = client, request
        self._expected = expected_output_tokens
        self.completion: Completion | None = None
        self.first_token_s: float | None = None  # from the start of the call to the first text

    def __iter__(self):
        client, request = self._client, self._request
        key = cache_key({"provider": client._provider, **request})
        if client._cache is not None and (hit := client._cache.get(key)) is not None:
            self.completion = Completion(**hit, cached=True)
            self.first_token_s = 0.0
            if hit["text"]:
                yield hit["text"]
            return
        model, messages = request["model"], request["messages"]
        throttle = client._throttle(model)
        wait_s = throttle.acquire(
            estimate_tokens(messages, request["max_completion_tokens"], self._expected)
        )
        started = client._clock()
        chunks, headers, _ = client._send_with_retry(request, client._transport.send_stream)
        throttle.update(headers)
        parts: list[str] = []
        finish, usage, served_model = "", {}, model
        for chunk in chunks:
            served_model = chunk.get("model") or served_model
            usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage
            for choice in chunk.get("choices") or []:
                text = (choice.get("delta") or {}).get("content") or ""
                finish = choice.get("finish_reason") or finish
                if text:
                    if self.first_token_s is None:
                        self.first_token_s = round(client._clock() - started, 4)
                    parts.append(text)
                    yield text
        details = usage.get("completion_tokens_details") or {}
        record = {
            "text": "".join(parts),
            "model": served_model,
            "finish_reason": finish,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": details.get("reasoning_tokens", 0),
            },
            "latency_s": round(client._clock() - started, 4),
        }
        if client._cache is not None:
            client._cache.put(key, record)
        self.completion = Completion(**record, wait_s=round(wait_s, 3))
