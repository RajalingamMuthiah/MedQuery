"""Bootstrap confidence intervals and Cohen's kappa (numpy only)."""

from collections.abc import Callable, Sequence

import numpy as np

N_RESAMPLES = 1000
SEED = 0


def bootstrap_ci(
    items: Sequence,
    metric: Callable[[Sequence], float | None],
    n: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = 0.95,
) -> tuple[float | None, float | None, float | None]:
    """(point estimate, low, high): percentile bootstrap over items, resampled with replacement.

    `metric` recomputes the statistic from a list of items, so pooled ratios are recomputed per
    resample. Resamples where the metric is undefined (None) are skipped."""
    point = metric(items)
    if point is None or not items:
        return point, None, None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        sample = [items[i] for i in rng.integers(0, len(items), len(items))]
        value = metric(sample)
        if value is not None:
            values.append(value)
    if not values:
        return point, None, None
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return point, float(low), float(high)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa for two raters over the same units. None if undefined (no units)."""
    if len(a) != len(b):
        raise ValueError("raters labelled different numbers of units")
    if not a:
        return None
    labels = sorted(set(a) | set(b))
    index = {label: i for i, label in enumerate(labels)}
    table = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b, strict=True):
        table[index[x], index[y]] += 1
    total = table.sum()
    observed = np.trace(table) / total
    expected = float((table.sum(axis=0) * table.sum(axis=1)).sum() / total**2)
    if expected == 1.0:  # both raters used one identical label throughout
        return 1.0 if observed == 1.0 else None
    return float((observed - expected) / (1 - expected))


def agreement(a: Sequence[str], b: Sequence[str]) -> float | None:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else None
