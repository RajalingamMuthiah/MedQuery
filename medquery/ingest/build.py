"""Ingest the corpus: verified PDFs -> section-aware chunks (JSONL) plus a build summary."""

import json
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from medquery.fileio import atomic_write, exclusive_lock
from medquery.ingest.blocks import Page
from medquery.ingest.chunk import SHORT_WORDS, Chunk, chunk_document
from medquery.ingest.download import verify
from medquery.ingest.manifest import Document, Manifest
from medquery.ingest.pdf import extract_pages

LOCK_NAME = ".build.lock"
MARGIN = 0.08  # top/bottom fraction of the page where running headers and footers live
MIN_REPEATS = 3
TOC_LINE_RE = re.compile(r"(\.{2,}|\s)\s*\d{1,3}$")  # "Chapter 2: Testing ..... 37"
TOC_MIN_LINES = 6


def is_toc_page(page: Page, follows_toc: bool = False) -> bool:
    """A table-of-contents page: most lines end in a page number, and it either says
    "contents" or directly follows a contents page (a continuation)."""
    lines = [line.text for box in page.boxes for line in box.lines]
    numbered = sum(1 for text in lines if TOC_LINE_RE.search(text))
    mentions_contents = any("contents" in text.lower() for text in lines)
    mostly_numbered = numbered >= TOC_MIN_LINES and numbered >= 0.5 * len(lines)
    return mostly_numbered and (mentions_contents or follows_toc)


def drop_toc_pages(pages: list[Page]) -> None:
    previous_was_toc = False
    for page in pages:
        is_toc = page.source == "text" and is_toc_page(page, previous_was_toc)
        if is_toc:
            page.boxes, page.tables = [], []
        previous_was_toc = is_toc


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def strip_running_text(pages: list[Page]) -> None:
    """Drop page numbers and headers/footers that repeat across pages."""

    def in_margin(box) -> bool:
        return box.bottom <= MARGIN or box.top >= 1 - MARGIN

    def key(box) -> str:
        return _margin_key(" ".join(line.text for line in box.lines))

    counts = Counter(key(box) for page in pages for box in page.boxes if in_margin(box))
    for page in pages:
        page.boxes = [
            box
            for box in page.boxes
            if not (
                in_margin(box)
                and (counts[key(box)] >= MIN_REPEATS or re.fullmatch(r"[#ivxlc\s.-]*", key(box)))
            )
        ]


def record(doc: Document, chunk: Chunk) -> dict:
    return {
        "id": chunk.id,
        "doc_id": doc.id,
        "title": doc.title,
        "publisher": doc.publisher,
        "licence": doc.licence,
        "topic": doc.topic,
        "as_of": doc.as_of,
        "superseded": doc.superseded,
        **{k: v for k, v in asdict(chunk).items() if k != "id"},
    }


def doc_stats(doc: Document, pages: list[Page], chunks: list[Chunk], seconds: float) -> dict:
    ocr_pages = [p for p in pages if p.source == "ocr"]
    ocr_words = sum(p.ocr_words for p in ocr_pages)
    mean_conf = (
        round(sum(p.ocr_confidence * p.ocr_words for p in ocr_pages) / ocr_words, 1)
        if ocr_words
        else None
    )
    return {
        "doc_id": doc.id,
        "pages": len(pages),
        "text_pages": len(pages) - len(ocr_pages),
        "ocr_pages": len(ocr_pages),
        "tables": sum(len(p.tables) for p in pages),
        "chunks": len(chunks),
        "table_chunks": sum(1 for c in chunks if c.kind == "table"),
        "quiz_chunks": sum(1 for c in chunks if c.kind == "quiz"),
        "ocr_words": ocr_words,
        "ocr_mean_confidence": mean_conf,
        "seconds": round(seconds, 1),
    }


def build(manifest: Manifest, raw_dir: Path, out_dir: Path) -> dict:
    """Build chunks under an exclusive lock; every output file is replaced atomically.

    Raises LockHeld if another build holds `out_dir/.build.lock`.
    """
    with exclusive_lock(out_dir / LOCK_NAME):
        return _build(manifest, raw_dir, out_dir)


def _build(manifest: Manifest, raw_dir: Path, out_dir: Path) -> dict:
    docs = []
    text_word_counts: list[int] = []
    with (
        atomic_write(out_dir / "chunks.jsonl") as out,
        atomic_write(out_dir / "dropped.jsonl") as audit,
    ):
        for doc in manifest.documents:
            pdf_path = raw_dir / f"{doc.id}.pdf"
            if not pdf_path.exists():
                raise FileNotFoundError(f"{pdf_path} is missing; run the download command first")
            verify(pdf_path, doc.sha256)
            started = time.perf_counter()
            pages = extract_pages(pdf_path)
            strip_running_text(pages)
            drop_toc_pages(pages)
            dropped: list[tuple[str, Chunk]] = []
            chunks = chunk_document(doc.id, pages, dropped)
            for chunk in chunks:
                out.write(json.dumps(record(doc, chunk), ensure_ascii=False) + "\n")
                if chunk.kind == "text":
                    text_word_counts.append(len(chunk.text.split()))
            for reason, chunk in dropped:  # kept for auditing what the junk filter removed
                row = {"reason": reason, **record(doc, chunk)}
                audit.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats = doc_stats(doc, pages, chunks, time.perf_counter() - started)
            stats["dropped_junk"] = dict(Counter(reason for reason, _ in dropped))
            docs.append(stats)

    ocr_words = sum(d["ocr_words"] for d in docs)
    totals = {
        key: sum(d[key] for d in docs)
        for key in (
            "pages",
            "text_pages",
            "ocr_pages",
            "tables",
            "chunks",
            "table_chunks",
            "quiz_chunks",
        )
    }
    totals["ocr_mean_confidence"] = (
        round(sum((d["ocr_mean_confidence"] or 0) * d["ocr_words"] for d in docs) / ocr_words, 1)
        if ocr_words
        else None
    )
    dropped_total: Counter[str] = Counter()
    for d in docs:
        dropped_total.update(d["dropped_junk"])
    totals["dropped_junk"] = dict(dropped_total)
    short = sum(1 for n in text_word_counts if n < SHORT_WORDS)
    totals["short_text_chunks"] = short
    totals["short_text_share"] = (
        round(short / len(text_word_counts), 3) if text_word_counts else 0.0
    )
    summary = {"documents": docs, "totals": totals}
    with atomic_write(out_dir / "build_summary.json") as f:
        f.write(json.dumps(summary, indent=2) + "\n")
    return summary
