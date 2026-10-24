"""Failure diagnosis of a dev run: every incorrect/partial answer, unsupported claim and false
abstention, joined with hand-assigned causes from a committed CSV, plus table-chunk statistics.
No API calls."""

import csv
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from medquery.eval.report import REPORTS_DIR
from medquery.fileio import atomic_write

CAUSES_CSV = Path(__file__).parent / "diagnosis" / "phase4_causes.csv"
CAUSES = {
    "table_units": "table/units misread",
    "number_qualifier": "number/qualifier near-miss",
    "multi_passage": "multi-passage synthesis",
    "wrong_citation": "citation pointing to the wrong passage",
    "other_added_detail": "other: detail not in the passages",
    "other_paraphrase": "other: paraphrase beyond the passage",
    "retrieval_miss": "retrieval: gold not in the top 10",
}


class DiagnosisError(ValueError):
    pass


def findings(records: list[dict]) -> list[dict]:
    """One row per incorrect/partial answer, unsupported claim and false abstention."""
    out = []
    for r in sorted(records, key=lambda r: r["id"]):
        if r["answerable"] and r["abstained"]:
            out.append(
                {
                    "item_id": r["id"],
                    "finding": "false_abstention",
                    "claim_index": "",
                    "text": r["question"],
                    "type": r["type"],
                }
            )
            continue
        if not r["judge"]:
            continue
        if r["answerable"] and r["judge"]["correctness"] in ("incorrect", "partial"):
            out.append(
                {
                    "item_id": r["id"],
                    "finding": r["judge"]["correctness"],
                    "claim_index": "",
                    "text": r["answer"],
                    "type": r["type"],
                }
            )
        for i, claim in enumerate(r["judge"]["claims"]):
            if not claim["supported_by_context"]:
                out.append(
                    {
                        "item_id": r["id"],
                        "finding": "unsupported_claim",
                        "claim_index": str(i),
                        "text": claim["text"],
                        "type": r["type"],
                    }
                )
    return out


def load_causes(run_id: str, path: Path = CAUSES_CSV) -> dict[tuple, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["run_id"] == run_id]
    bad = [r for r in rows if r["cause"] not in CAUSES]
    if bad:
        raise DiagnosisError(f"unknown cause(s): {sorted({r['cause'] for r in bad})}")
    return {(r["item_id"], r["finding"], r["claim_index"]): r for r in rows}


def join(found: list[dict], causes: dict[tuple, dict]) -> list[dict]:
    """Every finding must have exactly one cause row, and every cause row a finding."""
    keys = {(f["item_id"], f["finding"], f["claim_index"]) for f in found}
    missing = sorted(keys - set(causes))
    extra = sorted(set(causes) - keys)
    if missing or extra:
        raise DiagnosisError(
            f"findings without a cause: {missing}; causes without a finding: {extra}"
        )
    return [f | causes[(f["item_id"], f["finding"], f["claim_index"])] for f in found]


def diagnosis_markdown(
    run_id: str, config_name: str, rows: list[dict], stats: dict, git: dict, created: str
) -> str:
    dirty = "unknown" if git["dirty"] is None else ("DIRTY" if git["dirty"] else "clean")
    by_cause = Counter(r["cause"] for r in rows)
    lines = [
        f"# MedQuery failure diagnosis: run `{run_id}` (`{config_name}`, dev)",
        "",
        f"- Created (UTC): {created}",
        f"- Git commit: `{git['commit']}` ({dirty} working tree)",
        f"- Causes: hand-assigned in `medquery/eval/diagnosis/{CAUSES_CSV.name}`; the findings "
        "come from the run's judge verdicts. No API calls.",
        "",
        "## Findings by cause",
        "",
        "| cause | findings | items |",
        "|---|---|---|",
    ]
    for cause, label in CAUSES.items():
        items = sorted({r["item_id"] for r in rows if r["cause"] == cause})
        if by_cause[cause]:
            lines.append(f"| {label} | {by_cause[cause]} | {', '.join(items)} |")
    lines += [
        "",
        "## Every finding",
        "",
        "| item | type | finding | claim | cause | note |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        text = r["text"].replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {r['item_id']} | {r['type']} | {r['finding']} | {text[:160]} | "
            f"{CAUSES[r['cause']]} | {r['note'].replace('|', '/')} |"
        )
    lines += [
        "",
        "## Table chunks (corpus v1)",
        "",
        "| measure | chunks |",
        "|---|---|",
        *[f"| {k.replace('_', ' ')} | {v} |" for k, v in stats.items()],
        "",
        "Continuation = a captionless table on the page after a table of the same document with "
        "the same number of columns. The table_v2 fix targets every row of this table.",
    ]
    return "\n".join(lines) + "\n"


def write_diagnosis(run_id, config_name, rows, stats, git, out_dir: Path = REPORTS_DIR) -> Path:
    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    path = out_dir / f"{created[:10]}_diagnosis_dev_{run_id}.md"
    with atomic_write(path) as f:
        f.write(diagnosis_markdown(run_id, config_name, rows, stats, git, created))
    return path
