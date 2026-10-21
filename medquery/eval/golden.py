"""Golden-set items: validation rules and the hand-review CSV.

Evidence is anchored as (doc_id, page, exact quote), never as chunk ids (see evidence.py).
"""

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from medquery import paths
from medquery.eval.evidence import Evidence, EvidenceMapper, normalise
from medquery.fileio import atomic_write

GOLDEN_DIR = paths.ROOT / "medquery" / "eval" / "golden"
TYPES = ("factual", "table", "multi-chunk", "superseded", "unanswerable")
DECISIONS = ("keep", "fix", "drop")
NGRAM = 6  # a question sharing this many consecutive words with its source page is rejected
QUOTE_WORDS = (4, 30)
QUESTION_WORDS = (6, 45)
DUPLICATE_JACCARD = 0.7

REVIEW_COLUMNS = [
    "id",
    "decision",
    "reviewed",
    "question",
    "reference_answer",
    "evidence",
    "type",
    "topic",
    "answerable",
    "source",
    "generator_note",
    "notes",
]
_EVIDENCE_LINE = re.compile(r'^\s*(\S+)\s*\|\s*(\d+)\s*\|\s*"(.*)"\s*$')


@dataclass
class Item:
    id: str
    question: str
    reference_answer: str
    evidence: list[Evidence]
    type: str
    topic: str
    answerable: bool
    source_pages: list[tuple[str, int]] = field(default_factory=list)
    gen_model: str = ""
    generator_note: str = ""


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.'][a-z0-9]+)*", normalise(text))


def shared_ngram(question: str, source: str, n: int = NGRAM) -> str | None:
    """The first run of n consecutive words the question copies from the source, if any."""
    q, s = words(question), words(source)
    source_grams = {tuple(s[i : i + n]) for i in range(len(s) - n + 1)}
    for i in range(len(q) - n + 1):
        if tuple(q[i : i + n]) in source_grams:
            return " ".join(q[i : i + n])
    return None


def copied_spans(question: str, source: str, n: int = NGRAM) -> list[str]:
    """Every maximal run of at least n consecutive question words that also appears in the
    source (overlapping n-grams merged), for telling a rewrite exactly what to avoid."""
    q, s = words(question), words(source)
    source_grams = {tuple(s[i : i + n]) for i in range(len(s) - n + 1)}
    copied = [False] * len(q)
    for i in range(len(q) - n + 1):
        if tuple(q[i : i + n]) in source_grams:
            for j in range(i, i + n):
                copied[j] = True
    spans, current = [], []
    for word, flag in zip(q, copied, strict=True):
        if flag:
            current.append(word)
        elif current:
            spans.append(" ".join(current))
            current = []
    if current:
        spans.append(" ".join(current))
    return spans


def is_duplicate(question: str, accepted: list[str]) -> bool:
    mine = set(words(question))
    for other in accepted:
        theirs = set(words(other))
        if mine and len(mine & theirs) / len(mine | theirs) >= DUPLICATE_JACCARD:
            return True
    return False


def check_question(question: str) -> str | None:
    n = len(question.split())
    if not question.strip().endswith("?"):
        return "question does not end with '?'"
    if not QUESTION_WORDS[0] <= n <= QUESTION_WORDS[1]:
        return f"question has {n} words (allowed {QUESTION_WORDS[0]}-{QUESTION_WORDS[1]})"
    return None


def check_evidence(
    evidence: list[Evidence], page_texts: dict[tuple[str, int], str], mapper: EvidenceMapper
) -> str | None:
    """Every quote must be on its stated page (normalised) and map to at least one chunk."""
    if not 1 <= len(evidence) <= 3:
        return f"{len(evidence)} evidence quotes (allowed 1-3)"
    for ev in evidence:
        n = len(ev.quote.split())
        if not QUOTE_WORDS[0] <= n <= QUOTE_WORDS[1]:
            return f"quote has {n} words (allowed {QUOTE_WORDS[0]}-{QUOTE_WORDS[1]})"
        page = page_texts.get((ev.doc_id, ev.page))
        if page is None:
            return f"evidence cites {ev.doc_id} p.{ev.page}, which is not a source page"
        if normalise(ev.quote) not in normalise(page):
            return f"quote not found verbatim on {ev.doc_id} p.{ev.page}: {ev.quote[:60]!r}"
        if mapper.map(ev).status == "missing":
            return f"quote is on the page but in no chunk: {ev.quote[:60]!r}"
    return None


def format_evidence(evidence: list[Evidence]) -> str:
    return "\n".join(f'{ev.doc_id} | {ev.page} | "{ev.quote}"' for ev in evidence)


def parse_evidence(cell: str) -> list[Evidence]:
    evidence = []
    for line in cell.splitlines():
        if not line.strip():
            continue
        match = _EVIDENCE_LINE.match(line)
        if not match:
            raise ValueError(
                f'evidence line must look like: doc_id | page | "quote"  (got {line!r})'
            )
        evidence.append(Evidence(match[1], int(match[2]), match[3]))
    return evidence


def review_row(item: Item, source: str) -> dict:
    return {
        "id": item.id,
        "decision": "keep",
        "reviewed": "",
        "question": item.question,
        "reference_answer": item.reference_answer,
        "evidence": format_evidence(item.evidence),
        "type": item.type,
        "topic": item.topic,
        "answerable": "yes" if item.answerable else "no",
        "source": source,
        "generator_note": item.generator_note,
        "notes": "",
    }


def write_review_csv(path: Path, rows: list[dict]) -> None:
    """UTF-8 with a BOM so Excel shows symbols such as ≥ and µ correctly."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    with atomic_write(path) as f:
        f.write("﻿" + buffer.getvalue())


def read_review_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
