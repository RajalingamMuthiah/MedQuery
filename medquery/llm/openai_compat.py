"""Transport for OpenAI-compatible chat APIs (Groq), via the official openai SDK."""

import openai
from openai import OpenAI

from medquery.llm.base import TransportError


class OpenAICompatTransport:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 120.0):
        # max_retries=0: retries are handled (and tested) in LLMClient, not hidden in the SDK.
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout_s)
        self._base_url = base_url

    def __repr__(self) -> str:
        return f"OpenAICompatTransport(base_url={self._base_url!r})"

    def send(self, request: dict) -> tuple[dict, dict]:
        try:
            raw = self._client.chat.completions.with_raw_response.create(**request)
        except openai.APIStatusError as exc:
            raise TransportError(
                exc.status_code, dict(exc.response.headers), _error_message(exc)
            ) from None
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise TransportError(None, {}, type(exc).__name__) from None
        return raw.parse().model_dump(), dict(raw.headers)


def _error_message(exc: openai.APIStatusError) -> str:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body) if isinstance(body, dict) else {}
    return str(error.get("message") or exc.message) if isinstance(error, dict) else exc.message
