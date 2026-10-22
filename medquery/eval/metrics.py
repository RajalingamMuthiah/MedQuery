"""Metric definitions over per-item eval records, each with a 95% bootstrap CI.

Every function takes a list of item records and returns a value or None (undefined), so the
bootstrap can recompute it on resamples. Eligibility (answerable/answered) is applied inside.
"""

from dataclasses import dataclass

import numpy as np

from medquery.eval.judge import CORRECTNESS
from medquery.eval.stats import bootstrap_ci

RECALL_KS = (1, 3, 5, 10)


def _answerable_with_gold(items):
    return [i for i in items if i["answerable"] and i["gold_chunks"]]


def _answered_judged(items):
    return [i for i in items if not i["abstained"] and i.get("judge")]


def _mean(values):
    return float(np.mean(values)) if values else None


def recall_at(k: int):
    def metric(items):
        rows = _answerable_with_gold(items)
        return _mean(
            [
                len(set(i["gold_chunks"]) & set(i["retrieved_ids"][:k])) / len(i["gold_chunks"])
                for i in rows
            ]
        )

    return metric


def mrr(items):
    values = []
    for i in _answerable_with_gold(items):
        gold = set(i["gold_chunks"])
        rank = next((r for r, cid in enumerate(i["retrieved_ids"][:10], 1) if cid in gold), None)
        values.append(1 / rank if rank else 0.0)
    return _mean(values)


def context_precision(items):
    """Rank-aware average precision over the k passages given to the generator:
    (1 / min(|gold|, k)) * sum over relevant ranks r of precision@r."""
    values = []
    for i in _answerable_with_gold(items):
        gold, context = set(i["gold_chunks"]), i["context_ids"]
        hits, total = 0, 0.0
        for rank, cid in enumerate(context, 1):
            if cid in gold:
                hits += 1
                total += hits / rank
        values.append(total / min(len(gold), len(context)) if context else 0.0)
    return _mean(values)


def context_recall(items):
    """Share of gold evidence items with at least one mapped chunk among the context passages."""
    values = []
    for i in _answerable_with_gold(items):
        mapped = [e for e in i["evidence_map"] if e["chunk_ids"]]
        if mapped:
            covered = sum(1 for e in mapped if set(e["chunk_ids"]) & set(i["context_ids"]))
            values.append(covered / len(mapped))
    return _mean(values)


def abstention_accuracy(items):
    return _mean([float(i["abstained"] != i["answerable"]) for i in items])


def abstention_recall(items):
    """Share of unanswerable questions where the system abstained."""
    return _mean([float(i["abstained"]) for i in items if not i["answerable"]])


def false_abstention_rate(items):
    """Share of answerable questions where the system wrongly abstained."""
    return _mean([float(i["abstained"]) for i in items if i["answerable"]])


def answer_correctness(items):
    """correct 1, partial 0.5, incorrect 0; abstaining on an answerable question scores 0.
    Items whose judge failed are excluded (and counted in the report)."""
    values = []
    for i in items:
        if not i["answerable"]:
            continue
        if i["abstained"]:
            values.append(0.0)
        elif i.get("judge"):
            values.append(CORRECTNESS[i["judge"]["correctness"]])
    return _mean(values)


def faithfulness(items):
    """Per answer: claims supported by the retrieved passages / claims; averaged over answers."""
    return _mean(
        [
            sum(c["supported_by_context"] for c in i["judge"]["claims"]) / len(i["judge"]["claims"])
            for i in _answered_judged(items)
            if i["judge"]["claims"]
        ]
    )


def hallucination_rate(items):
    """Pooled: unsupported claims / all claims, over all judged answers."""
    claims = [c for i in _answered_judged(items) for c in i["judge"]["claims"]]
    return sum(not c["supported_by_context"] for c in claims) / len(claims) if claims else None


def answer_relevancy(items):
    return _mean([(i["judge"]["relevancy"] - 1) / 4 for i in _answered_judged(items)])


def citation_accuracy(items):
    """Pooled: cited claims whose cited passages support them / cited claims."""
    cited = [c for i in _answered_judged(items) for c in i["judge"]["claims"] if c["cited_ids"]]
    return sum(bool(c["supported_by_cited"]) for c in cited) / len(cited) if cited else None


def citation_validity(items):
    """Pooled: citations that name a passage actually given to the generator / all citations."""
    pairs = [(cid, i) for i in items if not i["abstained"] for cid in i["cited_ids"]]
    return sum(cid in i["context_ids"] for cid, i in pairs) / len(pairs) if pairs else None


def uncited_claim_share(items):
    claims = [c for i in _answered_judged(items) for c in i["judge"]["claims"]]
    return sum(not c["cited_ids"] for c in claims) / len(claims) if claims else None


@dataclass(frozen=True)
class MetricDef:
    name: str
    fn: object
    over: str
    definition: str


QUALITY_METRICS = [
    MetricDef(
        "answer_correctness",
        answer_correctness,
        "answerable",
        "judge vs reference: correct 1, partial 0.5, incorrect 0; abstained = 0",
    ),
    MetricDef(
        "faithfulness",
        faithfulness,
        "answered",
        "per answer, claims supported by retrieved passages / claims; mean",
    ),
    MetricDef(
        "hallucination_rate",
        hallucination_rate,
        "answered",
        "unsupported claims / all claims, pooled",
    ),
    MetricDef("answer_relevancy", answer_relevancy, "answered", "judge rating 1-5 scaled to 0-1"),
    MetricDef(
        "context_precision",
        context_precision,
        "answerable",
        "rank-aware AP@k over the generator's passages; relevant = maps to a gold quote",
    ),
    MetricDef(
        "context_recall",
        context_recall,
        "answerable",
        "gold evidence items covered by the generator's passages",
    ),
    *[
        MetricDef(f"recall@{k}", recall_at(k), "answerable", f"gold chunks in the top {k} / gold")
        for k in RECALL_KS
    ],
    MetricDef("mrr", mrr, "answerable", "1 / rank of the first gold chunk in the top 10"),
    MetricDef(
        "citation_accuracy",
        citation_accuracy,
        "answered",
        "cited claims whose cited passages support them / cited claims, pooled",
    ),
    MetricDef(
        "citation_validity",
        citation_validity,
        "answered",
        "citations naming a passage given to the generator / citations",
    ),
    MetricDef("uncited_claim_share", uncited_claim_share, "answered", "claims with no citation"),
    MetricDef(
        "abstention_accuracy",
        abstention_accuracy,
        "all",
        "abstain on unanswerable, answer on answerable",
    ),
    MetricDef(
        "abstention_recall",
        abstention_recall,
        "unanswerable",
        "unanswerable questions where the system abstained",
    ),
    MetricDef(
        "false_abstention_rate",
        false_abstention_rate,
        "answerable",
        "answerable questions where the system abstained",
    ),
]


def compute(items: list[dict]) -> list[dict]:
    rows = []
    for m in QUALITY_METRICS:
        point, low, high = bootstrap_ci(items, m.fn)
        rows.append(
            {
                "metric": m.name,
                "value": point,
                "ci_low": low,
                "ci_high": high,
                "over": m.over,
                "definition": m.definition,
            }
        )
    return rows


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None}
    p50, p95 = np.percentile(values, [50, 95])
    return {"p50": float(p50), "p95": float(p95)}
