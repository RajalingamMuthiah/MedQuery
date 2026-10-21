"""Token and list-price cost accounting for a set of LLM calls."""

from collections import defaultdict

from medquery import config
from medquery.llm.base import Completion


def summarize(completions: list[Completion]) -> dict[str, dict]:
    """Per model: calls, cache hits, tokens and list cost, both for the whole artifact
    (including cached calls, at their original usage) and for calls made in this run."""
    out: dict[str, dict] = defaultdict(
        lambda: {
            "calls": 0,
            "cached": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "cost_this_run_usd": 0.0,
        }
    )
    for c in completions:
        row = out[c.model]
        cost = config.cost_usd(
            _price_key(c.model),
            c.usage.get("prompt_tokens", 0),
            c.usage.get("completion_tokens", 0),
        )
        row["calls"] += 1
        row["cached"] += int(c.cached)
        row["prompt_tokens"] += c.usage.get("prompt_tokens", 0)
        row["completion_tokens"] += c.usage.get("completion_tokens", 0)
        row["cost_usd"] += cost
        if not c.cached:
            row["cost_this_run_usd"] += cost
    return dict(out)


def _price_key(model: str) -> str:
    """Responses may name the model slightly differently; fall back to a prefix match."""
    if model in config.PRICES:
        return model
    for known in config.PRICES:
        if model.startswith(known) or known.endswith(model):
            return known
    raise KeyError(f"no list price recorded for model {model!r} in medquery/config.py")


def format_summary(summary: dict[str, dict]) -> list[str]:
    lines = []
    for model, row in sorted(summary.items()):
        lines.append(
            f"  {model:<22} {row['calls']:>4} calls ({row['cached']} cached)  "
            f"{row['prompt_tokens']:>8} in / {row['completion_tokens']:>7} out tokens  "
            f"${row['cost_usd']:.4f} list (${row['cost_this_run_usd']:.4f} this run)"
        )
    return lines
