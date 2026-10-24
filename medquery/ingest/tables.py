"""Table chunks: quality statistics (Phase 4 diagnosis). The table_v2 fixes are added here too."""

import re

_RULE_RE = re.compile(r"^\|[-| :]+\|$")
_SPLIT_DECIMAL_RE = re.compile(r"\d \.\d")
_CAPTION_END_RE = re.compile(r"[.:)]\s*$")


def _rows(text: str) -> list[list[str]]:
    rows = [line for line in text.splitlines() if line.startswith("|") and not _RULE_RE.match(line)]
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]


def caption(text: str) -> str:
    """The first non-table line of a table chunk ('' if the chunk starts with the table)."""
    for line in text.splitlines():
        if line.strip() and not line.startswith("|"):
            return line.strip()
    return ""


def is_continuation(previous: dict, chunk: dict) -> bool:
    """A captionless table on the page after a table of the same document with the same
    number of columns: most likely the same table continued."""
    if previous["kind"] != "table" or chunk["kind"] != "table":
        return False
    if previous["doc_id"] != chunk["doc_id"] or caption(chunk["text"]):
        return False
    if chunk["page_start"] != previous["page_end"] + 1:
        return False
    rows_a, rows_b = _rows(previous["text"]), _rows(chunk["text"])
    return bool(rows_a and rows_b) and len(rows_a[0]) == len(rows_b[0])


def table_stats(chunks: list[dict]) -> dict:
    """Counts of the table-chunk problems the table_v2 fix targets."""
    tables = [c for c in chunks if c["kind"] == "table"]
    stats = {
        "chunks": len(chunks),
        "table_chunks": len(tables),
        "no_caption": 0,
        "caption_cut_mid_sentence": 0,
        "continuation_without_caption_or_header": 0,
        "multi_row_header": 0,
        "all_empty_columns": 0,
        "split_decimals": 0,
    }
    for c in tables:
        cap = caption(c["text"])
        if not cap:
            stats["no_caption"] += 1
        elif not _CAPTION_END_RE.search(cap):
            stats["caption_cut_mid_sentence"] += 1
        rows = _rows(c["text"])
        if rows:
            width = len(rows[0])
            if any(all(i >= len(r) or not r[i] for r in rows) for i in range(width)):
                stats["all_empty_columns"] += 1
            second = rows[1] if len(rows) > 1 else []
            if any(second) and not any(re.search(r"\d", cell) for cell in second[1:]):
                stats["multi_row_header"] += 1
        if _SPLIT_DECIMAL_RE.search(c["text"]):
            stats["split_decimals"] += 1
    stats["continuation_without_caption_or_header"] = sum(
        is_continuation(a, b) for a, b in zip(chunks, chunks[1:], strict=False)
    )
    return stats
