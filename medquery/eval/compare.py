"""Ablation across retrieval configs on the SAME questions: paired bootstrap differences, a
failure diagnosis per config, a multi-chunk breakout and reranker truncation."""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from medquery import config as settings
from medquery.eval import metrics
from medquery.eval.report import REPORTS_DIR, _ci, _fmt
from medquery.eval.runner import RunConfig
from medquery.eval.stats import N_RESAMPLES, SEED, bootstrap_ci
from medquery.fileio import atomic_write
from medquery.retrieval.rerank import RERANKERS

PAIRED = [
    ("answer_correctness", metrics.answer_correctness),
    ("recall@5", metrics.recall_at(5)),
    ("mrr", metrics.mrr),
    ("false_abstention_rate", metrics.false_abstention_rate),
    # Guards: better retrieval must not make the system answer unanswerable questions or
    # cite less accurately.
    ("abstention_recall", metrics.abstention_recall),
    ("citation_accuracy", metrics.citation_accuracy),
]
LOWER_IS_BETTER = {"false_abstention_rate"}
SUMMARY = [
    ("answer_correctness", metrics.answer_correctness),
    ("faithfulness", metrics.faithfulness),
    ("hallucination_rate", metrics.hallucination_rate),
    ("context_recall", metrics.context_recall),
    ("recall@5", metrics.recall_at(5)),
    ("recall@10", metrics.recall_at(10)),
    ("mrr", metrics.mrr),
    ("abstention_accuracy", metrics.abstention_accuracy),
    ("false_abstention_rate", metrics.false_abstention_rate),
    ("abstention_recall", metrics.abstention_recall),
    ("citation_accuracy", metrics.citation_accuracy),
]
# Rows added after looking at dev results, labelled as such in the report.
POST_HOC = {
    name: "post-hoc fix, decided after seeing dev reranker truncation"
    for name in (
        "hybrid_rerank_maxp",
        "hybrid_rerank_headers_maxp",
        "hybrid_rerank_maxp_k8",
        "hybrid_rerank_headers_maxp_k8",
    )
}
TRUNCATION_NOTE = 0.10  # note it in the report above this share of reranker inputs


def paired_bootstrap(
    previous: Sequence[dict],
    current: Sequence[dict],
    metric: Callable[[Sequence[dict]], float | None],
    n: int = N_RESAMPLES,
    seed: int = SEED,
) -> tuple[float | None, float | None, float | None]:
    """(current - previous) on the same questions, with a 95% percentile bootstrap CI. Each
    resample draws question ids once and uses them for both configs."""
    a = {r["id"]: r for r in previous}
    b = {r["id"]: r for r in current}
    if set(a) != set(b):
        raise ValueError("configs were evaluated on different questions; cannot pair them")
    ids = sorted(a)

    def diff(index) -> float | None:
        va = metric([a[ids[i]] for i in index])
        vb = metric([b[ids[i]] for i in index])
        return None if va is None or vb is None else vb - va

    point = diff(range(len(ids)))
    if point is None:
        return None, None, None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        d = diff(rng.integers(0, len(ids), len(ids)))
        if d is not None:
            values.append(d)
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def _same_except(a: RunConfig, b: RunConfig, **fields) -> bool:
    return asdict(replace(a, name="", **fields)) == asdict(replace(b, name=""))


def comparison_partner(index: int, runs: list) -> int:
    """Which earlier row row `index` is compared with: a k = 8 row with the same config at
    k = 5; a MaxP row with the same config truncating instead of windowing; otherwise the
    previous row (the cumulative ablation)."""
    cfg = runs[index][0]
    for j in range(index):
        other = runs[j][0]
        if cfg.k != other.k and _same_except(cfg, other, k=other.k):
            return j
        if cfg.rerank_window and _same_except(cfg, other, rerank_window=0, rerank_overlap=0):
            return j
    return index - 1


BEST_ROW_RULE = (
    "highest answer correctness on dev; ties broken by the lower false-abstention rate "
    "(point estimates)"
)


def best_row(runs: list) -> tuple[int, list[dict]]:
    """Index of the best row among those at the default k, by BEST_ROW_RULE, and the values
    the choice was made on (one dict per candidate, in run order)."""
    table = []
    for index, (cfg, _, records) in enumerate(runs):
        if cfg.k != settings.TOP_K:
            continue
        table.append(
            {
                "index": index,
                "config": cfg.name,
                "answer_correctness": metrics.answer_correctness(records),
                "false_abstention_rate": metrics.false_abstention_rate(records),
            }
        )
    if not table:
        raise ValueError(f"no row at k = {settings.TOP_K} to choose from")
    best = min(
        table,
        key=lambda t: (-round(t["answer_correctness"], 9), round(t["false_abstention_rate"], 9)),
    )
    return best["index"], table


def gold_position(record: dict, k: int) -> str:
    """Where the best-ranked gold chunk sat: in the generator's k passages, retrieved but
    beyond k (within the top 10), or not in the top 10."""
    gold = set(record["gold_chunks"])
    rank = next((n for n, cid in enumerate(record["retrieved_ids"][:10], 1) if cid in gold), None)
    if rank is None:
        return "not in top 10"
    return "in context" if rank <= k else "beyond k"


def diagnose(records: list[dict], k: int) -> dict[str, dict[str, int]]:
    false_abstentions = [r for r in records if r["answerable"] and r["abstained"]]
    wrong = [
        r
        for r in records
        if r["answerable"]
        and not r["abstained"]
        and r["judge"]
        and r["judge"]["correctness"] in ("incorrect", "partial")
    ]
    return {
        "false abstention": dict(Counter(gold_position(r, k) for r in false_abstentions)),
        "incorrect/partial": dict(Counter(gold_position(r, k) for r in wrong)),
    }


def multi_chunk(records: list[dict]) -> dict:
    rows = [r for r in records if r["type"] == "multi-chunk"]
    out = {"n": len(rows)}
    for name, fn in [
        ("answer_correctness", metrics.answer_correctness),
        ("context_recall", metrics.context_recall),
        ("recall@5", metrics.recall_at(5)),
        ("recall@10", metrics.recall_at(10)),
    ]:
        out[name] = bootstrap_ci(rows, fn)
    out["both_gold_in_context"] = sum(
        1 for r in rows if r["gold_chunks"] and set(r["gold_chunks"]) <= set(r["context_ids"])
    )
    return out


def truncation(records: list[dict]) -> dict | None:
    stats = [r["retrieval_stats"] for r in records if r.get("retrieval_stats")]
    if not stats:
        return None
    pairs = sum(s["rerank_pairs"] for s in stats)
    cut = sum(s["rerank_truncated"] for s in stats)
    return {"pairs": pairs, "truncated": cut, "share": cut / pairs if pairs else 0.0}


def _rerank_input(cfg: RunConfig) -> str:
    if not cfg.rerank_model:
        return "-"
    if cfg.rerank_window:
        return f"MaxP: {cfg.rerank_window}-token windows, overlap {cfg.rerank_overlap}"
    return "whole passage (truncated at 512)"


def _latency(records: list[dict]) -> dict:
    return {
        "retrieval_ms": {
            k: (v * 1000 if v is not None else None)
            for k, v in metrics.percentiles([r["latency"]["retrieval_s"] for r in records]).items()
        },
        "end_to_end_s": metrics.percentiles([r["latency"]["end_to_end_s"] for r in records]),
        "generation_cost_per_query": float(np.mean([r["cost_usd"]["generation"] for r in records])),
        "judge_cost_total": float(sum(r["cost_usd"]["judge"] for r in records)),
    }


def ablation_markdown(
    runs: list[tuple[RunConfig, str, list[dict]]],
    split: str,
    git: dict,
    golden_meta: dict,
    chunks_sha: str,
    created: str,
) -> tuple[str, list[dict]]:
    dirty = "unknown" if git["dirty"] is None else ("DIRTY" if git["dirty"] else "clean")
    lines = [
        f"# MedQuery ablation: retrieval upgrades on the **{split}** split",
        "",
        f"- Created (UTC): {created}",
        f"- Git commit: `{git['commit']}` ({dirty} working tree)",
        f"- Golden set: `{golden_meta['version']}` sha256 `{golden_meta['sha256']}`; corpus "
        f"`chunks.jsonl` sha256 `{chunks_sha}`",
        f"- Fixed across rows: generator `{runs[0][0].generator}` with "
        f"`{runs[0][0].answer_prompt}`, judge `{runs[0][0].judge}` with "
        f"`{runs[0][0].judge_prompt}`; list prices: {settings.price_note()}",
        f"- Paired bootstrap: {N_RESAMPLES} resamples of the same question ids, seed {SEED}; "
        "a difference counts as **real** only if its 95% CI excludes 0.",
        "",
        "## Configurations",
        "",
        "| row | config | run id | retriever | RRF k | reranker (licence) | rerank N "
        "| rerank input | headers | k | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, (cfg, rid, _) in enumerate(runs, 1):
        reranker = (
            f"`{cfg.rerank_model}` ({RERANKERS.get(cfg.rerank_model, '?')})"
            if cfg.rerank_model
            else "none"
        )
        lines.append(
            f"| {n} | `{cfg.name}` | `{rid}` | {cfg.retriever} | "
            f"{cfg.fuse_k if cfg.retriever == 'hybrid' else '-'} | {reranker} | "
            f"{cfg.rerank_n if cfg.rerank_model else '-'} | {_rerank_input(cfg)} | "
            f"{cfg.headers} | {cfg.k} | {POST_HOC.get(cfg.name, '')} |"
        )

    csv_rows: list[dict] = []
    lines += [
        "",
        "## Metrics (95% bootstrap CI)",
        "",
        "| config | " + " | ".join(name for name, _ in SUMMARY) + " |",
        "|---|" + "---|" * len(SUMMARY),
    ]
    for cfg, _, records in runs:
        cells = []
        for name, fn in SUMMARY:
            value, low, high = bootstrap_ci(records, fn)
            cells.append(f"{_fmt(value)} {_ci(low, high)}")
            csv_rows.append(
                {"config": cfg.name, "metric": name, "value": value, "ci_low": low, "ci_high": high}
            )
        lines.append(f"| `{cfg.name}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Paired differences",
        "",
        "Each row against the previous row, except: a k = 8 row against the same config at "
        "k = 5, and a MaxP row against the same config with truncated reranker input.",
        "",
        "| config vs comparison row | metric | difference | 95% CI | real? |",
        "|---|---|---|---|---|",
    ]
    for index in range(1, len(runs)):
        prev_cfg, _, prev = runs[comparison_partner(index, runs)]
        cfg, _, cur = runs[index]
        for name, fn in PAIRED:
            d, low, high = paired_bootstrap(prev, cur, fn)
            real = "n/a" if low is None else ("**yes**" if low > 0 or high < 0 else "no")
            lines.append(
                f"| `{cfg.name}` vs `{prev_cfg.name}` | {name} | "
                f"{'n/a' if d is None else f'{d:+.3f}'} | {_ci(low, high)} | {real} |"
            )
            csv_rows.append(
                {
                    "config": cfg.name,
                    "metric": f"diff_vs:{prev_cfg.name}:{name}",
                    "value": d,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    lines.append(
        "For false_abstention_rate a negative difference is an improvement; for every other "
        "metric, a positive one. abstention_recall (unanswerable questions where the system "
        "abstained) and citation_accuracy are guards: a real drop in either means better "
        "retrieval made the system answer unanswerable questions or cite less accurately."
    )

    lines += [
        "",
        "## Multi-chunk questions",
        "",
        "Questions whose answer needs two passages. Context recall counts the gold "
        "evidence covered by all k passages the generator saw, so it shows whether a "
        "larger k recovers the second gold chunk.",
        "",
        "| config | n | answer correctness | context recall | recall@5 | recall@10 | "
        "both gold in context |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg, _, records in runs:
        m = multi_chunk(records)
        cells = [
            f"{_fmt(m[k][0])} {_ci(m[k][1], m[k][2])}"
            for k in ("answer_correctness", "context_recall", "recall@5", "recall@10")
        ]
        lines.append(
            f"| `{cfg.name}` | {m['n']} | "
            + " | ".join(cells)
            + f" | {m['both_gold_in_context']} of {m['n']} |"
        )

    larger = [cfg for cfg, _, _ in runs if cfg.k != settings.TOP_K]
    if larger:
        chosen, table = best_row(runs)
        best_cfg = runs[chosen][0]
        lines += [
            "",
            f"## Choice of the k = {larger[0].k} row",
            "",
            f"Rule, stated before the k = {larger[0].k} run: {BEST_ROW_RULE}. Candidates are "
            f"all rows at k = {settings.TOP_K}, including post-hoc ones.",
            "",
            "| config | answer correctness | false-abstention rate | chosen |",
            "|---|---|---|---|",
        ]
        for t in table:
            lines.append(
                f"| `{t['config']}` | {_fmt(t['answer_correctness'])} | "
                f"{_fmt(t['false_abstention_rate'])} | "
                f"{'**yes**' if t['index'] == chosen else ''} |"
            )
        for cfg in larger:
            same = _same_except(cfg, best_cfg, k=best_cfg.k)
            lines.append(
                f"`{cfg.name}` is "
                + ("the chosen row at k = " if same else "**NOT** the chosen row at k = ")
                + f"{cfg.k}."
            )

    lines += [
        "",
        "## Failure diagnosis",
        "",
        "Where the best-ranked gold chunk sat, relative to the k passages given to the "
        "generator. Misses outside the context are retrieval failures; failures with "
        "gold in context point at generation (Phase 4).",
        "",
        "| config | false abstentions | incorrect/partial |",
        "|---|---|---|",
    ]
    for cfg, _, records in runs:
        d = diagnose(records, cfg.k)
        lines.append(f"| `{cfg.name}` | {d['false abstention']} | {d['incorrect/partial']} |")

    lines += [
        "",
        "## Reranker input truncation",
        "",
        "Inputs scored are (question, passage) pairs; for a MaxP row, (question, window) "
        "pairs. Truncated = over the model's 512-token limit.",
        "",
        "| config | inputs scored | over max length (truncated) | share |",
        "|---|---|---|---|",
    ]
    notes = []
    for cfg, _, records in runs:
        t = truncation(records)
        if t is None:
            lines.append(f"| `{cfg.name}` | - | - | - |")
            continue
        lines.append(f"| `{cfg.name}` | {t['pairs']} | {t['truncated']} | {t['share']:.1%} |")
        if t["share"] > TRUNCATION_NOTE:
            notes.append(
                f"**Note:** {t['share']:.0%} of `{cfg.name}` reranker inputs exceed the "
                "model's max length and are truncated, so the reranker never sees the "
                "end of those passages."
            )
    lines += [""] + notes if notes else []

    lines += [
        "",
        "## Latency and cost",
        "",
        "| config | retrieval p50 / p95 (ms) | end-to-end p50 / p95 (s) | generation cost "
        "per query | judge cost (eval overhead) |",
        "|---|---|---|---|---|",
    ]
    for cfg, _, records in runs:
        lat = _latency(records)
        r, e = lat["retrieval_ms"], lat["end_to_end_s"]
        lines.append(
            f"| `{cfg.name}` | {_fmt(r['p50'], 1)} / {_fmt(r['p95'], 1)} | "
            f"{_fmt(e['p50'], 2)} / {_fmt(e['p95'], 2)} | "
            f"${lat['generation_cost_per_query']:.6f} | ${lat['judge_cost_total']:.4f} |"
        )
    lines.append(
        "Retrieval latency is measured live (query embedding, BM25, fusion, rerank); "
        "generation latency is the original API call's, from the cache."
    )
    return "\n".join(lines) + "\n", csv_rows


def write_ablation(runs, split, git, golden_meta, chunks_sha, out_dir: Path = REPORTS_DIR):
    import csv
    import hashlib
    import io

    if split != "dev":
        raise ValueError("Phase 3 ablations run on the dev split only")
    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    md, csv_rows = ablation_markdown(runs, split, git, golden_meta, chunks_sha, created)
    key = hashlib.sha256("|".join(rid for _, rid, _ in runs).encode()).hexdigest()[:10]
    stem = f"{created[:10]}_ablation_{split}_{key}"
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=["config", "metric", "value", "ci_low", "ci_high"], lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(csv_rows)
    paths = [out_dir / f"{stem}.md", out_dir / f"{stem}.csv"]
    for path, text in zip(paths, (md, buffer.getvalue()), strict=True):
        with atomic_write(path) as f:
            f.write(text)
    return paths


def config_summary(cfg: RunConfig) -> dict:
    return asdict(cfg)
