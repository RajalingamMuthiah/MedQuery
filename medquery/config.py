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
PRICE_SOURCES = {
    "groq": ("https://console.groq.com/docs/models", "2026-10-21"),
    "openai": ("https://developers.openai.com/api/docs/pricing", "2026-10-22"),
}
PRICES_SOURCE = PRICE_SOURCES["groq"][0]  # kept for older call sites
PRICES_DATE = PRICE_SOURCES["groq"][1]

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5


def provider_of(model: str) -> str:
    if model not in MODEL_PROVIDER:
        raise KeyError(f"no provider configured for model {model!r} in medquery/config.py")
    return MODEL_PROVIDER[model]


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """List-price cost of one call (completion tokens include reasoning tokens)."""
    price_in, price_out = PRICES[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def price_note() -> str:
    return "; ".join(f"{p}: {url} ({day})" for p, (url, day) in PRICE_SOURCES.items())
