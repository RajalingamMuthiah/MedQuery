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
    ("hallucination_rate", metrics.hallucination_rate),  # primary (Phase 4 lesson)
    # Guards: better retrieval must not make the system answer unanswerable questions or
    # cite less accurately.
    ("abstention_recall", metrics.abstention_recall),
    ("citation_accuracy", metrics.citation_accuracy),
]
LOWER_IS_BETTER = {"false_abstention_rate", "hallucination_rate"}
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


def verifier_guards(records: list[dict], base: list[dict]) -> dict:
    """How the verifier changed answers, against the same items without it (`base`): sentences
    removed; over-removal = removed sentences whose matched judge claims (the judge's verdict on
    the unverified answer) were all supported; missed = judge-unsupported claims left in the
    verified answers."""
    from medquery.eval.judge_review import matching_parts

    by_id = {r["id"]: r for r in base}
    out = {
        "verified": 0,
        "failed": 0,
        "changed": 0,
        "to_abstention": 0,
        "removed": 0,
        "removed_supported": 0,
        "removed_unsupported": 0,
        "removed_unmatched": 0,
        "missed": 0,
    }
    for r in records:
        v = r.get("verification")
        if v is None:
            continue
        out["verified"] += 1
        out["failed"] += v["failed"]
        out["changed"] += bool(v["removed"])
        out["to_abstention"] += bool(v["removed"]) and r["abstained"]
        claims = (by_id.get(r["id"], {}).get("judge") or {}).get("claims", [])
        for sentence in v["removed"]:
            out["removed"] += 1
            parts = matching_parts(sentence, claims)
            if not parts:
                out["removed_unmatched"] += 1
            elif all(p["supported_by_context"] for p in parts):
                out["removed_supported"] += 1
            else:
                out["removed_unsupported"] += 1
        if r.get("judge"):
            out["missed"] += sum(not c["supported_by_context"] for c in r["judge"]["claims"])
    return out


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
    title: str = "retrieval upgrades",
    row_notes: dict[str, str] | None = None,
    extra_lines: list[str] | None = None,
    reparse: bool = False,
    appendix: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """row_notes: per-config note for the configs table (e.g. how an offline row was made);
    extra_lines: added to the header list (e.g. the Phase 4 budget); reparse: re-parse
    citations from the stored answers for every row and report citation validity."""
    row_notes = row_notes or {}
    summary = SUMMARY
    if reparse:
        runs = [(cfg, rid, reparse_citations(records)) for cfg, rid, records in runs]
        summary = [*SUMMARY, ("citation_validity", metrics.citation_validity)]
    dirty = "unknown" if git["dirty"] is None else ("DIRTY" if git["dirty"] else "clean")
    prompts = sorted({cfg.answer_prompt for cfg, _, _ in runs})
    lines = [
        f"# MedQuery ablation: {title} on the **{split}** split",
        "",
        f"- Created (UTC): {created}",
        f"- Git commit: `{git['commit']}` ({dirty} working tree)",
        f"- Golden set: `{golden_meta['version']}` sha256 `{golden_meta['sha256']}`; corpus "
        f"`chunks.jsonl` sha256 {chunks_sha if ' ' in chunks_sha else f'`{chunks_sha}`'}",
        f"- Generator `{runs[0][0].generator}`; answer prompt(s) "
        + ", ".join(f"`{p}`" for p in prompts)
        + f"; judge `{runs[0][0].judge}` with `{runs[0][0].judge_prompt}`; list prices: "
        f"{settings.price_note()}",
        f"- Paired bootstrap: {N_RESAMPLES} resamples of the same question ids, seed {SEED}; "
        "a difference counts as **real** only if its 95% CI excludes 0.",
        *[f"- {line}" for line in extra_lines or []],
        *(
            ["- Citations re-parsed from the stored answers with the current parser for every row."]
            if reparse
            else []
        ),
        "",
        "## Configurations",
        "",
        "| row | config | run id | retriever | RRF k | reranker (licence) | rerank N "
        "| rerank input | headers | k | generator provider | corpus | answer prompt | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
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
            f"{cfg.headers} | {cfg.k} | {cfg.generator_provider} | {cfg.corpus} | "
            f"`{cfg.answer_prompt.split(':')[0]}` | "
            f"{row_notes.get(cfg.name, POST_HOC.get(cfg.name, ''))} |"
        )

    csv_rows: list[dict] = []
    lines += [
        "",
        "## Metrics (95% bootstrap CI)",
        "",
        "| config | " + " | ".join(name for name, _ in summary) + " |",
        "|---|" + "---|" * len(summary),
    ]
    for cfg, _, records in runs:
        cells = []
        for name, fn in summary:
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
        "For false_abstention_rate and hallucination_rate a negative difference is an "
        "improvement; for every other "
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
    verified = [i for i, (_, _, recs) in enumerate(runs) if any("verification" in r for r in recs)]
    if verified:
        lines += [
            "",
            "## Verifier guards",
            "",
            "Over-removal = removed sentences whose matched judge claims (the judge's verdict on "
            "the same answer before verification) were all supported. Missed = claims the judge "
            "marks unsupported in the verified answers.",
            "",
            "| config vs base | answers verified | verifier failures | answers changed | turned "
            "into abstentions | sentences removed | removed but judge-supported (over-removal) | "
            "removed and judge-unsupported | removed, no judge claim matched | missed |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i in verified:
            base_cfg, _, base = runs[comparison_partner(i, runs)]
            cfg, _, recs = runs[i]
            g = verifier_guards(recs, base)
            lines.append(
                f"| `{cfg.name}` vs `{base_cfg.name}` | {g['verified']} | {g['failed']} | "
                f"{g['changed']} | {g['to_abstention']} | {g['removed']} | "
                f"{g['removed_supported']} | {g['removed_unsupported']} | "
                f"{g['removed_unmatched']} | {g['missed']} |"
            )
            verify_s = [r["verification"]["verify_s"] for r in recs if r.get("verification")]
            p = metrics.percentiles(verify_s)
            lines.append(
                f"Verifier latency for `{cfg.name}`: p50 {_fmt(p['p50'], 2)} s, "
                f"p95 {_fmt(p['p95'], 2)} s per answer (included in end-to-end latency)."
            )
    lines += appendix or []
    return "\n".join(lines) + "\n", csv_rows


def write_ablation(
    runs, split, git, golden_meta, chunks_sha, out_dir: Path = REPORTS_DIR, **options
):
    """options: title, row_notes, extra_lines, reparse (see ablation_markdown)."""
    import csv
    import hashlib
    import io

    final = options.pop("final", False)
    if split != "dev" and not final:
        raise ValueError("ablations run on the dev split only; the test split needs final=True")
    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    md, csv_rows = ablation_markdown(runs, split, git, golden_meta, chunks_sha, created, **options)
    key = hashlib.sha256("|".join(rid for _, rid, _ in runs).encode()).hexdigest()[:10]
    stem = f"{created[:10]}_{'final' if final else 'ablation'}_{split}_{key}"
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


# --- Phase 4: provider check (same retrieval, generator on Groq vs NVIDIA) ---

PROVIDER_METRICS = [*SUMMARY, ("citation_validity", metrics.citation_validity)]


def reparse_citations(records: list[dict]) -> list[dict]:
    """Records with citations re-parsed from the stored answer text by the current parser, so
    rows generated before and after the parser fix are compared on the same terms."""
    from medquery.generation.prompts import extract_citations

    return [
        r if r["abstained"] else r | {"cited_ids": extract_citations(r["answer"])} for r in records
    ]


def _generation(records: list[dict]) -> dict:
    return {
        "generation_s": metrics.percentiles([r["latency"]["generation_s"] for r in records]),
        "truncated": sum(r["finish_reason"] == "length" for r in records),
        "generation_cost_per_query": float(np.mean([r["cost_usd"]["generation"] for r in records])),
        "judge_cost_total": float(sum(r["cost_usd"]["judge"] for r in records)),
    }


def provider_check_markdown(
    pairs: list, git: dict, golden_meta: dict, chunks_sha: str, created: str
):
    """pairs: (groq config, run id, records, nvidia config, run id, records). Returns the report,
    CSV rows and whether any paired difference is real (its 95% CI excludes 0)."""
    dirty = "unknown" if git["dirty"] is None else ("DIRTY" if git["dirty"] else "clean")
    first = pairs[0][0]
    lines = [
        "# MedQuery provider check: generator on Groq vs NVIDIA NIM (dev split)",
        "",
        f"- Created (UTC): {created}",
        f"- Git commit: `{git['commit']}` ({dirty} working tree)",
        f"- Golden set: `{golden_meta['version']}` sha256 `{golden_meta['sha256']}`; corpus "
        f"`chunks.jsonl` sha256 `{chunks_sha}`",
        f"- Same model `{first.generator}`, prompt `{first.answer_prompt}`, judge `{first.judge}` "
        f"with `{first.judge_prompt}`. Retrieval is identical within each pair, so differences "
        "come from generation only.",
        f"- Paired bootstrap: {N_RESAMPLES} resamples of the same question ids, seed {SEED}; a "
        "difference counts as **real** only if its 95% CI excludes 0.",
        "- Citations are re-parsed from the stored answers with the current parser (Unicode "
        "hyphens and fullwidth brackets normalised) for both providers.",
        f"- Prices: {settings.price_note()}",
    ]
    csv_rows, any_real = [], False
    for groq_cfg, groq_rid, groq_recs, _nv_cfg, nv_rid, nv_recs in pairs:
        groq_recs, nv_recs = reparse_citations(groq_recs), reparse_citations(nv_recs)
        lines += [
            "",
            f"## `{groq_cfg.name}`: Groq run `{groq_rid}` vs NVIDIA run `{nv_rid}`",
            "",
            "| metric | Groq | NVIDIA | NVIDIA - Groq | 95% CI | real? |",
            "|---|---|---|---|---|---|",
        ]
        for name, fn in PROVIDER_METRICS:
            g, n = fn(groq_recs), fn(nv_recs)
            d, low, high = paired_bootstrap(groq_recs, nv_recs, fn)
            real = low is not None and (low > 0 or high < 0)
            any_real |= real
            verdict = "n/a" if low is None else ("**yes**" if real else "no")
            diff = "n/a" if d is None else f"{d:+.3f}"
            lines.append(
                f"| {name} | {_fmt(g)} | {_fmt(n)} | {diff} | {_ci(low, high)} | {verdict} |"
            )
            csv_rows.append(
                {
                    "config": groq_cfg.name,
                    "metric": name,
                    "groq": g,
                    "nvidia": n,
                    "diff": d,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        gg, gn = _generation(groq_recs), _generation(nv_recs)
        lines += [
            "",
            "| | Groq | NVIDIA |",
            "|---|---|---|",
            f"| generation latency p50 / p95 (s) | {_fmt(gg['generation_s']['p50'], 2)} / "
            f"{_fmt(gg['generation_s']['p95'], 2)} | {_fmt(gn['generation_s']['p50'], 2)} / "
            f"{_fmt(gn['generation_s']['p95'], 2)} |",
            f"| truncated generations | {gg['truncated']} | {gn['truncated']} |",
            f"| generation cost per query | ${gg['generation_cost_per_query']:.6f} | "
            f"${gn['generation_cost_per_query']:.6f} |",
            f"| judge cost (eval overhead) | ${gg['judge_cost_total']:.4f} | "
            f"${gn['judge_cost_total']:.4f} |",
        ]
    verdict = (
        "at least one paired difference is real: the provider switch changes results."
        if any_real
        else "no paired difference is real: the provider switch is neutral on dev."
    )
    lines += ["", f"**Verdict:** {verdict}"]
    return "\n".join(lines) + "\n", csv_rows, any_real


def write_provider_check(pairs, split, git, golden_meta, chunks_sha, out_dir: Path = REPORTS_DIR):
    import csv
    import hashlib
    import io

    if split != "dev":
        raise ValueError("Phase 4 runs on the dev split only")
    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    md, rows, any_real = provider_check_markdown(pairs, git, golden_meta, chunks_sha, created)
    key = hashlib.sha256("|".join(p[1] + p[4] for p in pairs).encode()).hexdigest()[:10]
    stem = f"{created[:10]}_provider_check_{split}_{key}"
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["config", "metric", "groq", "nvidia", "diff", "ci_low", "ci_high"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    out = [out_dir / f"{stem}.md", out_dir / f"{stem}.csv"]
    for path, text in zip(out, (md, buffer.getvalue()), strict=True):
        with atomic_write(path) as f:
            f.write(text)
    return out, any_real


PRIMARY_METRICS = (
    "answer_correctness",
    "false_abstention_rate",
    "abstention_recall",
    "context_recall",
    "recall@5",
    "recall@10",
    "mrr",
)


def provider_check_section(csv_path: Path) -> list[str]:
    """Summary of a committed provider-check report, from its CSV: how many paired tests, which
    were real, and whether the primary metrics changed."""
    import csv

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["ci_low"] not in ("", "None")]
    real = [r for r in rows if float(r["ci_low"]) > 0 or float(r["ci_high"]) < 0]
    primary_real = [r for r in real if r["metric"] in PRIMARY_METRICS]
    lines = [
        "",
        "## Provider check (Groq vs NVIDIA)",
        "",
        f"From `{csv_path.stem}.md`: {len(rows)} paired tests (same retrieval, generator on Groq "
        f"vs NVIDIA). At 95% confidence about {len(rows) * 0.05:.1f} would look real by chance "
        "alone.",
        "",
    ]
    for r in real:
        direction = (
            "favours NVIDIA"
            if (float(r["diff"]) > 0) != (r["metric"] in LOWER_IS_BETTER)
            else "favours Groq"
        )
        lines.append(
            f"- Real: `{r['config']}` {r['metric']} {float(r['diff']):+.3f}, 95% CI "
            f"[{float(r['ci_low']):.4f}, {float(r['ci_high']):.3f}] ({direction}); expected by "
            f"chance across {len(rows)} paired tests."
        )
    lines.append(
        "- Primary metrics (answer correctness, false-abstention rate, abstention recall, "
        "retrieval): "
        + (
            "unchanged on every row."
            if not primary_real
            else f"{len(primary_real)} real difference(s)."
        )
    )
    lines.append(
        "- Decision (user, 2026-10-24): the switch is treated as neutral, and the Phase 4 rows "
        "run on NVIDIA."
    )
    return lines
