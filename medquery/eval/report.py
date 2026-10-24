"""Write the evaluation report: Markdown summary, metrics CSV and per-item CSV."""

import csv
import io
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from medquery import config as settings
from medquery import paths
from medquery.eval import metrics
from medquery.eval.runner import RunConfig
from medquery.fileio import atomic_write

REPORTS_DIR = paths.ROOT / "medquery" / "eval" / "reports"


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(low, high) -> str:
    return "n/a" if low is None else f"[{low:.3f}, {high:.3f}]"


def summarize(records: list[dict], config: RunConfig) -> dict:
    gen_cost = [r["cost_usd"]["generation"] for r in records]
    lat = [r["latency"] for r in records]
    mapping = Counter(e["status"] for r in records for e in r["evidence_map"])
    return {
        "metrics": metrics.compute(records),
        "counts": {
            "items": len(records),
            "answerable": sum(r["answerable"] for r in records),
            "unanswerable": sum(not r["answerable"] for r in records),
            "abstained": sum(r["abstained"] for r in records),
            "judged": sum(bool(r["judge"]) for r in records),
            "judge_failed": sum(r["judge_failed"] for r in records),
            "truncated_generations": sum(r["finish_reason"] == "length" for r in records),
            "answerable_without_gold_chunks": sum(
                r["answerable"] and not r["gold_chunks"] for r in records
            ),
        },
        "mapping": {s: mapping.get(s, 0) for s in ("exact", "fuzzy", "missing")},
        "correctness": dict(
            sorted(
                Counter(
                    "abstained"
                    if r["abstained"]
                    else (r["judge"] or {}).get("correctness", "judge_failed")
                    for r in records
                    if r["answerable"]
                ).items()
            )
        ),
        "latency": {
            "retrieval_ms": {
                k: v * 1000 if v is not None else None
                for k, v in metrics.percentiles([x["retrieval_s"] for x in lat]).items()
            },
            "generation_s": metrics.percentiles([x["generation_s"] for x in lat]),
            "end_to_end_s": metrics.percentiles([x["end_to_end_s"] for x in lat]),
            "throttle_wait_s_total": sum(x["wait_s"] for x in lat),
        },
        "cost": {
            "generation_per_query_usd": sum(gen_cost) / len(gen_cost) if gen_cost else None,
            "generation_total_usd": sum(gen_cost),
            "judge_total_usd": sum(r["cost_usd"]["judge"] for r in records),
            "generation_tokens": sum(
                r["generation_usage"]["prompt_tokens"] + r["generation_usage"]["completion_tokens"]
                for r in records
            ),
            "judge_tokens": sum(
                r["judge_usage"]["prompt_tokens"] + r["judge_usage"]["completion_tokens"]
                for r in records
            ),
        },
    }


def agreement_lines(block: dict | None) -> list[str]:
    if block is None:
        return [
            "**NOT YET VALIDATED.** The judge has not been checked against hand labels "
            "(judge_review_v1 pending), so the judge-based metrics (correctness, faithfulness, "
            "hallucination rate, relevancy, citation accuracy) are unverified."
        ]
    lines = []
    if block["stale"]:
        lines.append(
            f"**STALE:** labels were collected for {block['labelled_for']}, not the judge "
            "used in this run."
        )
    lines += [
        "| units | n (probes) | agreement [95% CI] | Cohen's kappa [95% CI] | human / judge |",
        "|---|---|---|---|---|",
    ]
    names = {
        "claim": "all claims",
        "claim_natural": "natural claims",
        "claim_probe": "mismatched-context probes",
        "answer": "answers",
    }
    for unit in ("claim", "claim_natural", "claim_probe", "answer"):
        if unit not in block:
            continue
        b = block[unit]
        a, k = b["agreement"], b["kappa"]
        lines.append(
            f"| {names[unit]} | {b['n']} ({b.get('n_probe', 0)}) | "
            f"{_fmt(a[0])} {_ci(a[1], a[2])} | "
            f"{_fmt(k[0])} {_ci(k[1], k[2])} | {b['human_class_counts']} / "
            f"{b['judge_class_counts']} |"
        )
    caught, rejected = block["natural_unsupported_caught"], block["natural_supported_rejected"]
    lines += [
        "",
        f"Natural claims: the judge caught **{caught['judge_caught']} of "
        f"{caught['human_unsupported']}** that the labeller marked unsupported, and wrongly "
        f"rejected **{rejected['judge_rejected']} of {rejected['human_supported']}** marked "
        "supported. With so few unsupported natural claims, kappa is unstable; read it together "
        "with these counts.",
        f"Labels: revision `{block.get('labels_revision', 'v1')}` by "
        f"{block.get('labeller', 'unknown')}, sha256 `{block['labels_sha256']}`"
        + (
            f"; supersedes `{block['supersedes']['file']}` (`{block['supersedes']['sha256']}`)"
            if block.get("supersedes")
            else ""
        )
        + ".",
    ]
    if block.get("unmatched"):
        lines.append(f"Units with no matching re-judged claim (excluded): {block['unmatched']}.")
    return lines


def history_lines(history: list[dict]) -> list[str]:
    """Every judge measured on the same frozen labels, for comparison across judge changes."""
    if not history:
        return []
    lines = [
        "",
        f"### Judge history on the same frozen labels (revision `{history[0]['labels_revision']}`)",
        "",
        "| judge model | prompt | natural unsupported caught | natural supported rejected | "
        "probe agreement | answer agreement | answer kappa |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in history:
        b = row["block"]
        caught, rejected = b["natural_unsupported_caught"], b["natural_supported_rejected"]
        lines.append(
            f"| `{row['judge_model']}` | `{row['judge_prompt']}` | "
            f"{caught['judge_caught']} of {caught['human_unsupported']} | "
            f"{rejected['judge_rejected']} of {rejected['human_supported']} | "
            f"{_fmt(b['claim_probe']['agreement'][0])} | {_fmt(b['answer']['agreement'][0])} | "
            f"{_fmt(b['answer']['kappa'][0])} |"
        )
    lines.append(
        "The judge_v2 prompt's exact-match rule was written after seeing these labels, so its "
        "measurement here is optimistic; it is re-validated on a fresh sample in Phase 3."
    )
    return lines


def markdown(
    summary: dict,
    config: RunConfig,
    split: str,
    run_id: str,
    golden_meta: dict,
    chunks_sha: str,
    git: dict,
    agreement: dict | None,
    created: str,
    history: list[dict] | None = None,
    notes: list[str] | None = None,
) -> str:
    c, lat, cost = summary["counts"], summary["latency"], summary["cost"]
    dirty = "unknown" if git["dirty"] is None else ("DIRTY" if git["dirty"] else "clean")
    lines = [
        f"# MedQuery eval: `{config.name}` on the **{split}** split",
        "",
        f"- Run date (UTC): {created}",
        f"- Run id: `{run_id}`",
        f"- Git commit: `{git['commit']}` ({dirty} working tree)",
        f"- Golden set: `{golden_meta['version']}` sha256 `{golden_meta['sha256']}` "
        f"({golden_meta['items']} items; reviewer: {golden_meta['reviewer']})",
        f"- Corpus: `data/processed/chunks.jsonl` sha256 `{chunks_sha}`",
        *[f"- {note}" for note in notes or []],
        "",
        "## Config",
        "",
        "| setting | value |",
        "|---|---|",
        *[f"| {k} | `{v}` |" for k, v in asdict(config).items()],
        f"| list prices | {settings.price_note()} |",
        "",
        "## Judge validation",
        "",
        *agreement_lines(agreement),
        *history_lines(history or []),
        "",
        "## Items",
        "",
        f"{c['items']} items: {c['answerable']} answerable, {c['unanswerable']} unanswerable. "
        f"Abstained: {c['abstained']}. Judged: {c['judged']}. Judge failures: {c['judge_failed']}. "
        f"Truncated generations: {c['truncated_generations']}.",
        "",
        f"Evidence mapping: {summary['mapping']['exact']} exact, {summary['mapping']['fuzzy']} "
        f"fuzzy, {summary['mapping']['missing']} missing quotes; answerable items with no gold "
        f"chunk: {c['answerable_without_gold_chunks']} (excluded from retrieval metrics).",
        "",
        "## Metrics (95% bootstrap CI, 1,000 resamples over questions, seed 0)",
        "",
        "| metric | value | 95% CI | over | definition |",
        "|---|---|---|---|---|",
        *[
            f"| {m['metric']} | {_fmt(m['value'])} | {_ci(m['ci_low'], m['ci_high'])} | "
            f"{m['over']} | {m['definition']} |"
            for m in summary["metrics"]
        ],
        "",
        f"Correctness on answerable items: {summary['correctness']}",
        "",
        "## Latency and cost",
        "",
        "| | p50 | p95 |",
        "|---|---|---|",
        f"| retrieval (ms) | {_fmt(lat['retrieval_ms']['p50'], 1)} | "
        f"{_fmt(lat['retrieval_ms']['p95'], 1)} |",
        f"| generation (s) | {_fmt(lat['generation_s']['p50'], 2)} | "
        f"{_fmt(lat['generation_s']['p95'], 2)} |",
        f"| end-to-end (s) | {_fmt(lat['end_to_end_s']['p50'], 2)} | "
        f"{_fmt(lat['end_to_end_s']['p95'], 2)} |",
        "",
        "Generation latency is the original API call's, kept in the cache, so re-runs report the "
        f"same figures. Client-side throttle waits (excluded above): "
        f"{lat['throttle_wait_s_total']:.1f} s in total.",
        "",
        f"Cost at list prices: generation ${_fmt(cost['generation_per_query_usd'], 6)} per query "
        f"(${_fmt(cost['generation_total_usd'], 4)} total, {cost['generation_tokens']} tokens); "
        f"judge ${_fmt(cost['judge_total_usd'], 4)} total ({cost['judge_tokens']} tokens, "
        "evaluation overhead, not part of per-query cost).",
    ]
    return "\n".join(lines) + "\n"


ITEM_COLUMNS = [
    "id",
    "split",
    "topic",
    "type",
    "answerable",
    "abstained",
    "correctness",
    "claims",
    "unsupported_claims",
    "relevancy",
    "cited_ids",
    "context_ids",
    "gold_chunks",
    "retrieval_ms",
    "generation_s",
    "judge_failed",
    "question",
    "answer",
]


def item_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        claims = (r["judge"] or {}).get("claims", [])
        rows.append(
            {
                "id": r["id"],
                "split": r["split"],
                "topic": r["topic"],
                "type": r["type"],
                "answerable": r["answerable"],
                "abstained": r["abstained"],
                "correctness": (r["judge"] or {}).get("correctness", ""),
                "claims": len(claims),
                "unsupported_claims": sum(not c["supported_by_context"] for c in claims),
                "relevancy": (r["judge"] or {}).get("relevancy", ""),
                "cited_ids": " ".join(r["cited_ids"]),
                "context_ids": " ".join(r["context_ids"]),
                "gold_chunks": " ".join(r["gold_chunks"]),
                "retrieval_ms": round(r["latency"]["retrieval_s"] * 1000, 1),
                "generation_s": r["latency"]["generation_s"],
                "judge_failed": r["judge_failed"],
                "question": r["question"],
                "answer": r["answer"],
            }
        )
    return rows


def _csv(rows: list[dict], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_report(
    records: list[dict],
    config: RunConfig,
    split: str,
    run_id: str,
    golden_meta: dict,
    chunks_sha: str,
    git: dict,
    agreement: dict | None,
    out_dir: Path = REPORTS_DIR,
    history: list[dict] | None = None,
    notes: list[str] | None = None,
) -> list[Path]:
    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    summary = summarize(records, config)
    stem = f"{created[:10]}_{config.name}_{split}_{run_id}"  # run id: never overwrite history
    md = markdown(
        summary,
        config,
        split,
        run_id,
        golden_meta,
        chunks_sha,
        git,
        agreement,
        created,
        history,
        notes,
    )
    written = []
    for path, text in (
        (out_dir / f"{stem}.md", md),
        (
            out_dir / f"{stem}.csv",
            _csv(
                summary["metrics"], ["metric", "value", "ci_low", "ci_high", "over", "definition"]
            ),
        ),
        (out_dir / f"{stem}_items.csv", _csv(item_rows(records), ITEM_COLUMNS)),
    ):
        with atomic_write(path) as f:
            f.write(text)
        written.append(path)
    return written
