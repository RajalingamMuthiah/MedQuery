"""LLM access behind a provider interface. Use get_client(); it routes each model to its
provider (Groq or OpenAI) as configured in medquery/config.py."""

import os

from dotenv import load_dotenv

from medquery import config, paths
from medquery.cache.disk import JsonCache
from medquery.llm.base import Completion, LLMError, QuotaExhausted
from medquery.llm.client import LLMClient
from medquery.llm.openai_compat import OpenAICompatTransport
from medquery.llm.throttle import Throttle

__all__ = ["Completion", "LLMClient", "LLMError", "QuotaExhausted", "Router", "get_client"]

LLM_CACHE = paths.ROOT / "data" / "cache" / "llm"


class Router:
    """Sends each call to the client of the model's provider; applies per-model overrides
    (e.g. omit temperature for models that only accept the default)."""

    def __init__(
        self,
        clients: dict[str, LLMClient],
        provider_of=config.provider_of,
        overrides: dict | None = None,
    ):
        self._clients = clients
        self._provider_of = provider_of
        self._overrides = config.MODEL_OVERRIDES if overrides is None else overrides

    def __repr__(self) -> str:
        return f"Router(providers={sorted(self._clients)})"

    def complete(self, messages: list[dict], *, model: str, **kwargs) -> Completion:
        provider = self._provider_of(model)
        if provider not in self._clients:
            key_env = config.PROVIDERS[provider]["key_env"]
            raise LLMError(
                f"{key_env} is not set: add it to .env to use {model} (see .env.example)"
            )
        kwargs = {**kwargs, **self._overrides.get(model, {})}
        return self._clients[provider].complete(messages, model=model, **kwargs)


def get_client() -> Router:
    """Clients for every provider whose key is in .env, with throttling, retries and the
    on-disk cache. Keys are never printed; each client redacts every configured key."""
    load_dotenv(paths.ROOT / ".env")
    keys = {name: os.environ.get(p["key_env"], "").strip() for name, p in config.PROVIDERS.items()}
    if not any(keys.values()):
        raise LLMError("no API keys set: add GROQ_API_KEY and/or OPENAI_API_KEY to .env")
    secrets = tuple(k for k in keys.values() if k)
    clients = {}
    for name, key in keys.items():
        if not key:
            continue
        provider = config.PROVIDERS[name]
        limits = provider["rate_limits"]
        clients[name] = LLMClient(
            OpenAICompatTransport(provider["base_url"], key),
            provider=name,
            cache=JsonCache(LLM_CACHE),
            throttle_factory=lambda limits=limits: Throttle(**limits),
            secrets=secrets,
        )
    return Router(clients)
