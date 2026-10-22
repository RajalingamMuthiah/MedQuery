"""Freeze a reviewed golden set: validate the review CSV, split dev/test, write golden_vN.

A frozen set is never edited. The runner loads it only if its sha256 matches the meta file.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from medquery.eval.evidence import EvidenceMapper, normalise
from medquery.eval.golden import (
    DECISIONS,
    GOLDEN_DIR,
    TYPES,
    check_question,
    parse_evidence,
    read_review_csv,
)
from medquery.fileio import atomic_write

MIN_REVIEWED = 0.25
TEST_SHARE = 0.30
SPLIT_SEED = 20261022
TOPICS = ("diabetes", "hypertension", "tuberculosis")


class FreezeError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__(f"{len(problems)} problem(s):\n  " + "\n  ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Frozen:
    items: list[dict]
    meta: dict


def _reviewed(value: str) -> bool:
    return value.strip().lower() in {"y", "yes"}


def validate_rows(
    rows: list[dict],
    page_texts: dict[tuple[str, int], str],
    mapper: EvidenceMapper,
    exclude: set[str],
) -> list[str]:
    problems = []
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids in the review file")
    for unknown in sorted(exclude - set(ids)):
        problems.append(f"--exclude names unknown id {unknown}")
    reviewed = sum(1 for r in rows if _reviewed(r["reviewed"]))
    if reviewed < MIN_REVIEWED * len(rows):
        problems.append(
            f"only {reviewed} of {len(rows)} rows are marked reviewed (need {MIN_REVIEWED:.0%})"
        )
    for r in rows:
        rid = r["id"]
        if r["decision"].strip() not in DECISIONS:
            problems.append(f"{rid}: decision {r['decision']!r} is not one of {DECISIONS}")
            continue
        if r["decision"].strip() == "drop" or rid in exclude:
            continue
        if r["type"] not in TYPES:
            problems.append(f"{rid}: unknown type {r['type']!r}")
        if r["topic"] not in TOPICS:
            problems.append(f"{rid}: unknown topic {r['topic']!r}")
        if r["answerable"] not in {"yes", "no"}:
            problems.append(f"{rid}: answerable must be yes or no")
            continue
        if reason := check_question(r["question"]):
            problems.append(f"{rid}: {reason}")
        try:
            evidence = parse_evidence(r["evidence"])
        except ValueError as exc:
            problems.append(f"{rid}: {exc}")
            continue
        answerable = r["answerable"] == "yes"
        if answerable != (r["type"] != "unanswerable"):
            problems.append(f"{rid}: type {r['type']} does not match answerable={r['answerable']}")
        if not answerable:
            if evidence:
                problems.append(f"{rid}: an unanswerable item must have no evidence")
            continue
        if not evidence:
            problems.append(f"{rid}: an answerable item needs at least one evidence quote")
        if not r["reference_answer"].strip():
            problems.append(f"{rid}: empty reference answer")
        for ev in evidence:  # re-checked for every kept item, not only fixed ones
            page = page_texts.get((ev.doc_id, ev.page))
            if page is None:
                problems.append(f"{rid}: no page {ev.doc_id} p.{ev.page}")
            elif normalise(ev.quote) not in normalise(page):
                problems.append(f"{rid}: quote not on {ev.doc_id} p.{ev.page}: {ev.quote[:60]!r}")
            elif mapper.map(ev).status == "missing":
                problems.append(f"{rid}: quote maps to no chunk: {ev.quote[:60]!r}")
    return problems


def assign_splits(items: list[dict], seed: int = SPLIT_SEED) -> None:
    """Stratified by (topic, type): about 30% of each stratum goes to test, deterministically."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in sorted(items, key=lambda i: i["id"]):
        strata[(item["topic"], item["type"])].append(item)
    for key in sorted(strata):
        members = strata[key]
        rng.shuffle(members)
        n_test = round(TEST_SHARE * len(members))
        for n, item in enumerate(members):
            item["split"] = "test" if n < n_test else "dev"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(
    review_path: Path,
    version: str,
    page_texts: dict[tuple[str, int], str],
    mapper: EvidenceMapper,
    reviewer: str,
    exclude: set[str] | None = None,
    candidates: list[dict] | None = None,
) -> Frozen:
    """Validate and build the frozen set (nothing is written; see write_frozen)."""
    exclude = exclude or set()
    rows = read_review_csv(review_path)
    if problems := validate_rows(rows, page_texts, mapper, exclude):
        raise FreezeError(problems)
    items = []
    for r in sorted(rows, key=lambda r: r["id"]):
        if r["decision"].strip() == "drop" or r["id"] in exclude:
            continue
        items.append(
            {
                "id": r["id"],
                "question": r["question"].strip(),
                "reference_answer": r["reference_answer"].strip(),
                "evidence": [
                    {"doc_id": e.doc_id, "page": e.page, "quote": e.quote}
                    for e in parse_evidence(r["evidence"])
                ],
                "type": r["type"],
                "topic": r["topic"],
                "answerable": r["answerable"] == "yes",
                "decision": r["decision"].strip(),
                "reviewed": _reviewed(r["reviewed"]),
                "source": r["source"],
                "notes": r["notes"],
            }
        )
    assign_splits(items)
    decisions = Counter(r["decision"].strip() for r in rows)
    reviewed = sum(1 for r in rows if _reviewed(r["reviewed"]))
    unanswerable = sum(1 for i in items if not i["answerable"])
    meta = {
        "version": version,
        "created": date.today().isoformat(),
        "reviewer": reviewer,
        "review_file": review_path.name,
        "review_file_sha256": sha256_file(review_path),
        "review": {
            "rows": len(rows),
            "reviewed": reviewed,
            "reviewed_share": round(reviewed / len(rows), 3),
            "keep": decisions["keep"],
            "fix": decisions["fix"],
            "drop": decisions["drop"],
            "excluded": sorted(exclude),
        },
        "items": len(items),
        "unanswerable": unanswerable,
        "unanswerable_share": round(unanswerable / len(items), 3),
        "by_topic_type": {
            f"{t}/{k}": n
            for (t, k), n in sorted(Counter((i["topic"], i["type"]) for i in items).items())
        },
        "by_split": dict(sorted(Counter(i["split"] for i in items).items())),
        "split": {"test_share": TEST_SHARE, "seed": SPLIT_SEED, "stratified_by": "topic,type"},
        # Every author model/prompt in the candidates file (accepted or rejected), not only the
        # ones behind frozen items: review rows do not carry their candidate's prompt.
        "candidate_authors": sorted(
            {c["gen_model"] for c in candidates or [] if c.get("gen_model")}
        ),
        "candidate_author_prompts": sorted(
            {c["gen_prompt"] for c in candidates or [] if c.get("gen_prompt")}
        ),
    }
    return Frozen(items, meta)


def frozen_paths(version: str, root: Path = GOLDEN_DIR) -> tuple[Path, Path]:
    return root / f"golden_{version}.jsonl", root / f"golden_{version}.meta.json"


def write_frozen(frozen: Frozen, root: Path = GOLDEN_DIR) -> tuple[Path, Path]:
    data_path, meta_path = frozen_paths(frozen.meta["version"], root)
    if data_path.exists() or meta_path.exists():
        raise FreezeError([f"{data_path.name} already exists: a frozen set is never overwritten"])
    with atomic_write(data_path) as f:
        for item in frozen.items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    meta = {**frozen.meta, "sha256": sha256_file(data_path)}
    with atomic_write(meta_path) as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return data_path, meta_path


def load_golden(version: str, root: Path = GOLDEN_DIR) -> tuple[list[dict], dict]:
    """Load a frozen set, refusing it if its sha256 does not match the meta file."""
    data_path, meta_path = frozen_paths(version, root)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual = sha256_file(data_path)
    if actual != meta["sha256"]:
        raise FreezeError([f"{data_path.name} sha256 {actual[:12]} != meta {meta['sha256'][:12]}"])
    items = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    return items, meta
