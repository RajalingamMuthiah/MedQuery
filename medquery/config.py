"""Model, provider, retrieval and pricing settings. Every report records the values it used."""

# Providers: base URL, API-key variable, and per-model client-side rate limits.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        # This account: TPM and RPD confirmed from response headers; RPM documented.
        "rate_limits": {"rpm": 30, "tpm": 8000},
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        # Headers show 10,000 RPM and 4M TPM for this key; we stay well below.
        "rate_limits": {"rpm": 500, "tpm": 400_000},
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        # Probe 2026-10-24: no rate-limit headers are returned, so this is a conservative
        # client-side limit, not a documented one.
        "rate_limits": {"rpm": 35, "tpm": 200_000},
    },
}
API_KEY_ENV = PROVIDERS["groq"]["key_env"]  # kept for older call sites

# Generator: production Groq model (live /models list and console.groq.com/docs/models,
# 2026-10-21). Judge: OpenAI gpt-6.1-sol (live /v1/models and developers.openai.com docs,
# 2026-10-22): current, not preview or deprecated, mid-cost; accepts seed and JSON mode but only
# the default temperature (1). gpt-4.1 was the only temperature-0 option and shuts down on
# 2027-01-14. Golden author: Groq preview Qwen, used once for the frozen golden set.
GENERATOR_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "gpt-6.1-sol"
GOLDEN_AUTHOR_MODEL = "qwen/qwen3.8-27b"
PREVIOUS_JUDGE_MODEL = "openai/gpt-oss-120b"  # Groq; kept for report history

# NVIDIA NIM (Phase 4 generator provider, chosen per run by RunConfig.generator_provider).
# Probe 2026-10-24: /v1/models lists "openai/gpt-oss-20b" (same id as Groq). The Groq request
# (temperature 0, seed 0, reasoning_effort "low", max_completion_tokens 600) returns 200, but so
# does an unknown parameter: NIM ignores what it does not support, so 200 proves nothing on its
# own. reasoning_effort is honoured ("high" used 135 completion tokens vs 20 for "low"); three
# repeats at temperature 0 / seed 0 were identical. Differences from Groq: no rate-limit headers,
# usage has no reasoning-token breakdown, and some calls queue for over 60 s.
MODEL_PROVIDER = {
    "openai/gpt-oss-20b": "groq",
    "openai/gpt-oss-120b": "groq",
    "qwen/qwen3.8-27b": "groq",
    "gpt-6.1-sol": "openai",
}
# Request parameters a model rejects: None means "omit and use the model's default".
MODEL_OVERRIDES = {"gpt-6.1-sol": {"temperature": None}}

# List prices in USD per 1M tokens (input, output), with source and date per provider.
PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.8-27b": (0.80, 4.00),
    "gpt-6.1-sol": (2.00, 10.00),
}
# Per-provider prices that differ from PRICES (same model id on another provider).
PROVIDER_PRICES = {
    ("nvidia", "openai/gpt-oss-20b"): (0.0, 0.0),  # API catalog free credits; no list price
}
PRICE_SOURCES = {
    "groq": ("https://console.groq.com/docs/models", "2026-10-21"),
    "openai": ("https://developers.openai.com/api/docs/pricing", "2026-10-22"),
    "nvidia": (
        "https://build.nvidia.com (API catalog free credits, no per-token list price)",
        "2026-10-24",
    ),
}
PRICES_SOURCE = PRICE_SOURCES["groq"][0]  # kept for older call sites
PRICES_DATE = PRICE_SOURCES["groq"][1]

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5


def provider_of(model: str) -> str:
    if model not in MODEL_PROVIDER:
        raise KeyError(f"no provider configured for model {model!r} in medquery/config.py")
    return MODEL_PROVIDER[model]


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int, provider: str = "") -> float:
    """List-price cost of one call (completion tokens include reasoning tokens). A provider
    with its own price for the model (PROVIDER_PRICES) overrides the model's list price."""
    price_in, price_out = PROVIDER_PRICES.get((provider, model), PRICES[model])
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def price_note() -> str:
    return "; ".join(f"{p}: {url} ({day})" for p, (url, day) in PRICE_SOURCES.items())
