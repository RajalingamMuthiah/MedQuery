"""LLM access behind a provider interface. Use get_client() rather than a provider directly."""

import os

from dotenv import load_dotenv

from medquery import config, paths
from medquery.cache.disk import JsonCache
from medquery.llm.base import Completion, LLMError, QuotaExhausted
from medquery.llm.client import LLMClient
from medquery.llm.openai_compat import OpenAICompatTransport
from medquery.llm.throttle import Throttle

__all__ = ["Completion", "LLMClient", "LLMError", "QuotaExhausted", "get_client"]

LLM_CACHE = paths.ROOT / "data" / "cache" / "llm"


def get_client() -> LLMClient:
    """A Groq client with throttling, retries and the on-disk cache. Reads GROQ_API_KEY."""
    load_dotenv(paths.ROOT / ".env")
    key = os.environ.get(config.API_KEY_ENV, "").strip()
    if not key:
        raise LLMError(f"{config.API_KEY_ENV} is not set: add it to .env (see .env.example)")
    return LLMClient(
        OpenAICompatTransport(config.GROQ_BASE_URL, key),
        provider="groq",
        cache=JsonCache(LLM_CACHE),
        throttle_factory=lambda: Throttle(**config.RATE_LIMITS),
        secrets=(key,),
    )
