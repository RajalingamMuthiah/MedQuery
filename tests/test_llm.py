import pytest

from medquery.cache.disk import JsonCache, cache_key
from medquery.llm.base import LLMError, QuotaExhausted, TransportError
from medquery.llm.client import MAX_TRIES, LLMClient
from medquery.llm.throttle import Throttle, estimate_tokens, parse_duration

SECRET = "gsk_test_secret_key_value_0123456789"
MESSAGES = [{"role": "user", "content": "What is LTBI?"}]


def ok(text: str = "answer") -> tuple[dict, dict]:
    data = {
        "model": "m",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }
    return data, {"x-ratelimit-remaining-tokens": "7000", "x-ratelimit-reset-tokens": "1s"}


class FakeTransport:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []

    def send(self, request: dict):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(transport, tmp_path=None, fake=None):
    fake = fake or FakeTime()
    return LLMClient(
        transport,
        provider="test",
        cache=JsonCache(tmp_path / "cache") if tmp_path else None,
        throttle_factory=lambda: Throttle(30, 8000, clock=fake.clock, sleep=fake.sleep),
        secrets=(SECRET,),
        sleep=fake.sleep,
        clock=fake.clock,
    ), fake


def test_successful_call_returns_text_and_usage():
    client, _ = make_client(FakeTransport(ok("hello")))
    result = client.complete(MESSAGES, model="m")
    assert result.text == "hello"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 2}
    assert not result.cached


def test_429_waits_for_retry_after_then_succeeds():
    transport = FakeTransport(TransportError(429, {"Retry-After": "7"}, "rate limited"), ok())
    client, fake = make_client(transport)
    assert client.complete(MESSAGES, model="m").text == "answer"
    assert fake.sleeps == [7.0]


def test_5xx_and_timeouts_back_off_exponentially():
    errors = [TransportError(503, {}, "unavailable"), TransportError(None, {}, "timeout")]
    client, fake = make_client(FakeTransport(*errors, ok()))
    client.complete(MESSAGES, model="m")
    assert len(fake.sleeps) == 2
    assert 1.0 <= fake.sleeps[0] < 1.3 and 2.0 <= fake.sleeps[1] < 2.6  # 1s, 2s plus jitter


def test_gives_up_after_max_tries():
    errors = [TransportError(500, {}, "boom") for _ in range(MAX_TRIES)]
    client, fake = make_client(FakeTransport(*errors))
    with pytest.raises(LLMError, match=f"gave up after {MAX_TRIES} tries"):
        client.complete(MESSAGES, model="m")
    assert len(fake.sleeps) == MAX_TRIES - 1


def test_client_errors_are_not_retried():
    client, fake = make_client(FakeTransport(TransportError(400, {}, "bad request")))
    with pytest.raises(LLMError, match="HTTP 400"):
        client.complete(MESSAGES, model="m")
    assert fake.sleeps == []


@pytest.mark.parametrize(
    "error",
    [
        TransportError(429, {"retry-after": "3600"}, "Rate limit reached"),
        TransportError(429, {}, "Limit 200000, Used 199990: tokens per day (TPD)"),
    ],
)
def test_daily_quota_stops_cleanly(error):
    client, fake = make_client(FakeTransport(error))
    with pytest.raises(QuotaExhausted, match="resume later"):
        client.complete(MESSAGES, model="m")
    assert fake.sleeps == []


def test_api_key_never_appears_in_errors_or_repr():
    leaky = TransportError(401, {}, f"Invalid API Key {SECRET}")
    client, _ = make_client(FakeTransport(leaky))
    with pytest.raises(LLMError) as info:
        client.complete(MESSAGES, model="m")
    assert SECRET not in str(info.value)
    assert "***" in str(info.value)
    assert SECRET not in repr(client)


def test_cache_hit_skips_the_transport_and_keeps_original_latency(tmp_path):
    transport = FakeTransport(ok("first"))
    client, _ = make_client(transport, tmp_path)
    first = client.complete(MESSAGES, model="m")
    second = client.complete(MESSAGES, model="m")
    assert len(transport.requests) == 1
    assert second.cached and second.text == "first"
    assert second.latency_s == first.latency_s


def test_any_parameter_change_misses_the_cache(tmp_path):
    transport = FakeTransport(ok("a"), ok("b"), ok("c"))
    client, _ = make_client(transport, tmp_path)
    client.complete(MESSAGES, model="m")
    client.complete(MESSAGES, model="m", temperature=0.5)
    client.complete(MESSAGES, model="other")
    assert len(transport.requests) == 3


def test_api_key_is_never_written_to_the_cache(tmp_path):
    client, _ = make_client(FakeTransport(ok()), tmp_path)
    client.complete(MESSAGES, model="m")
    for path in (tmp_path / "cache").rglob("*.json"):
        assert SECRET not in path.read_text(encoding="utf-8")


def test_request_parameters_are_passed_through():
    transport = FakeTransport(ok())
    client, _ = make_client(transport)
    schema = {"type": "json_object"}
    client.complete(
        MESSAGES, model="m", max_tokens=99, reasoning_effort="low", response_format=schema
    )
    request = transport.requests[0]
    assert request["max_completion_tokens"] == 99
    assert request["reasoning_effort"] == "low"
    assert request["response_format"] == schema
    assert request["seed"] == 0


def test_throttle_paces_requests_to_the_token_budget():
    fake = FakeTime()
    throttle = Throttle(rpm=30, tpm=8000, clock=fake.clock, sleep=fake.sleep)
    assert throttle.acquire(5000) == 0.0
    waited = throttle.acquire(5000)  # 10,000 > 8,000 tokens in the window: wait for it to roll
    assert waited == pytest.approx(60.0)


def test_throttle_paces_requests_to_the_request_budget():
    fake = FakeTime()
    throttle = Throttle(rpm=2, tpm=10**9, clock=fake.clock, sleep=fake.sleep)
    throttle.acquire(1)
    fake.now += 10
    throttle.acquire(1)
    assert throttle.acquire(1) == pytest.approx(50.0)


def test_throttle_honours_server_reset_when_budget_is_spent():
    fake = FakeTime()
    throttle = Throttle(rpm=30, tpm=8000, clock=fake.clock, sleep=fake.sleep)
    throttle.update({"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "4.5s"})
    assert throttle.acquire(10) == pytest.approx(4.5)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("1.065s", 1.065), ("1m26.4s", 86.4), ("450ms", 0.45), ("7", 7.0), ("2h3m", 7380.0)],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == pytest.approx(seconds)


def test_estimate_tokens_includes_output_budget():
    assert estimate_tokens([{"content": "x" * 350}], 100) == 200


def test_cache_key_ignores_key_order():
    assert cache_key({"a": 1, "b": [1, 2]}) == cache_key({"b": [1, 2], "a": 1})
