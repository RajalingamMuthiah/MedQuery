"""Abstention by retrieval confidence, computed offline from finished runs (no API calls).

Signal: the top-1 dense cosine similarity between the question and the passages given to the
generator. An answer whose signal is below the threshold becomes an abstention.

Operating point (fixed before looking, Phase 4 plan): the threshold that maximises the overall
dev score, where an answerable question scores its correctness (1 / 0.5 / 0; abstaining = 0)
and an unanswerable one scores 1 if the system abstains. Ties go to the lower threshold (more
answers). It is chosen and scored on the same questions, so it is optimistic; leave_one_out()
gives the honest estimate (threshold chosen on the other questions, applied to the held-out one).
"""

import math

from medquery.eval.judge import CORRECTNESS
from medquery.generation.prompts import ABSTAIN_TEXT

NO_THRESHOLD = -math.inf


def confidence(records: list[dict], chunks: dict[str, dict], embedder, index_text) -> dict:
    """{item id: max cosine(question, context passage)}, from local embeddings only."""
    import numpy as np

    from medquery.retrieval.embed import normalise

    ids = sorted({cid for r in records for cid in r["context_ids"]})
    texts = [index_text(chunks[cid]) for cid in ids]
    vectors = dict(zip(ids, normalise(embedder.embed_passages(texts)), strict=True))
    out = {}
    for r in records:
        q = normalise(np.asarray(embedder.embed_query(r["question"]), dtype=np.float32))
        sims = [float(np.dot(q, vectors[cid])) for cid in r["context_ids"]]
        out[r["id"]] = max(sims) if sims else 0.0
    return out


def credit(record: dict) -> float:
    if not record["answerable"]:
        return 1.0 if record["abstained"] else 0.0
    if record["abstained"] or not record.get("judge"):
        return 0.0
    return CORRECTNESS[record["judge"]["correctness"]]


def apply(records: list[dict], scores: dict, threshold: float) -> list[dict]:
    """Records with every answer whose signal is below `threshold` turned into an abstention."""
    out = []
    for r in records:
        if not r["abstained"] and scores[r["id"]] < threshold:
            r = r | {
                "abstained": True,
                "answer": ABSTAIN_TEXT,
                "cited_ids": [],
                "judge": None,
                "abstained_by_threshold": True,
            }
        out.append(r)
    return out


def overall(records: list[dict]) -> float:
    return sum(credit(r) for r in records) / len(records)


def candidates(scores: dict) -> list[float]:
    return [NO_THRESHOLD, *sorted(set(scores.values()))]


def curve(records: list[dict], scores: dict) -> list[dict]:
    """Risk-coverage points: for each threshold, the share of questions answered and the mean
    credit among answered questions (unanswerable ones answered score 0)."""
    points = []
    for t in candidates(scores):
        applied = apply(records, scores, t)
        answered = [r for r in applied if not r["abstained"]]
        points.append(
            {
                "threshold": t,
                "coverage": len(answered) / len(applied),
                "accuracy_answered": (sum(credit(r) for r in answered) / len(answered))
                if answered
                else None,
                "overall": overall(applied),
            }
        )
    return points


def operating_point(records: list[dict], scores: dict) -> float:
    best_t, best = NO_THRESHOLD, -1.0
    for t in candidates(scores):  # ascending, so ties keep the lower threshold
        value = overall(apply(records, scores, t))
        if value > best + 1e-12:
            best_t, best = t, value
    return best_t


def leave_one_out(records: list[dict], scores: dict) -> dict:
    """Each question scored with the threshold chosen on the other questions."""
    credits, thresholds = [], []
    for i, held in enumerate(records):
        rest = records[:i] + records[i + 1 :]
        t = operating_point(rest, {r["id"]: scores[r["id"]] for r in rest})
        thresholds.append(t)
        credits.append(credit(apply([held], scores, t)[0]))
    finite = [t for t in thresholds if t != NO_THRESHOLD]
    return {
        "overall": sum(credits) / len(credits),
        "thresholds_min": min(finite) if finite else None,
        "thresholds_max": max(finite) if finite else None,
        "no_threshold_folds": len(thresholds) - len(finite),
    }


def report_markdown(
    run_id: str, config_name: str, records: list[dict], scores: dict, git: dict, created: str
) -> str:
    dirty = "unknown" if git["dirty"] is None else ("DIRTY" if git["dirty"] else "clean")
    t = operating_point(records, scores)
    loo = leave_one_out(records, scores)
    applied = apply(records, scores, t)
    lines = [
        f"# MedQuery risk-coverage: run `{run_id}` (`{config_name}`, dev)",
        "",
        f"- Created (UTC): {created}",
        f"- Git commit: `{git['commit']}` ({dirty} working tree)",
        "- Signal: top-1 dense cosine similarity between the question and the passages given to "
        "the generator (local embeddings, no API calls). An answer below the threshold abstains.",
        "- Score: answerable = correctness (1 / 0.5 / 0; abstaining = 0); unanswerable = 1 if "
        "the system abstains. Overall = mean over all questions.",
        "- Operating point rule (fixed before looking): maximise the overall dev score; ties go "
        "to the lower threshold.",
        "",
        "## Operating point",
        "",
        "| | value |",
        "|---|---|",
        f"| threshold | {t:.4f} |" if t != NO_THRESHOLD else "| threshold | none (answer all) |",
        f"| overall score, no threshold | {overall(records):.3f} |",
        f"| overall score at the threshold (in-sample, optimistic) | {overall(applied):.3f} |",
        f"| overall score, leave-one-out (threshold chosen on the other questions) | "
        f"{loo['overall']:.3f} |",
        f"| answers turned into abstentions | "
        f"{sum(r.get('abstained_by_threshold', False) for r in applied)} |",
        f"| leave-one-out thresholds | {loo['thresholds_min']} to {loo['thresholds_max']} "
        f"({loo['no_threshold_folds']} folds chose none) |",
        "",
        "## Risk-coverage curve",
        "",
        "| threshold | coverage | accuracy among answered | overall |",
        "|---|---|---|---|",
    ]
    for p in curve(records, scores):
        th = "none" if p["threshold"] == NO_THRESHOLD else f"{p['threshold']:.4f}"
        acc = "n/a" if p["accuracy_answered"] is None else f"{p['accuracy_answered']:.3f}"
        lines.append(f"| {th} | {p['coverage']:.3f} | {acc} | {p['overall']:.3f} |")
    return "\n".join(lines) + "\n"


def write_report(run_id, config_name, records, scores, git):
    from datetime import UTC, datetime

    from medquery.eval.report import REPORTS_DIR
    from medquery.fileio import atomic_write

    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    path = REPORTS_DIR / f"{created[:10]}_risk_coverage_dev_{run_id}.md"
    with atomic_write(path) as f:
        f.write(report_markdown(run_id, config_name, records, scores, git, created))
    return path
