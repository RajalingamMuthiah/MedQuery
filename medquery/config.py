"""Model, retrieval and pricing settings. Every report records the values it used."""

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_ENV = "GROQ_API_KEY"

# Production models only for the RAG system and the judge (checked against the live /models
# list and https://console.groq.com/docs/models on 2026-10-21).
GENERATOR_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"
# Preview model, used once to author the frozen golden set (a different family from the judge).
GOLDEN_AUTHOR_MODEL = "qwen/qwen3.8-27b"

# Per-model limits for this account: TPM and RPD confirmed from response headers; RPM documented.
RATE_LIMITS = {"rpm": 30, "tpm": 8000}

# List prices in USD per 1M tokens (input, output).
PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}
PRICES_SOURCE = "https://console.groq.com/docs/models"
PRICES_DATE = "2026-10-21"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """List-price cost of one call (completion tokens include reasoning tokens)."""
    price_in, price_out = PRICES[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
