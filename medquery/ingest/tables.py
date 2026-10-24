"""Table chunks: quality statistics (Phase 4 diagnosis). The table_v2 fixes are added here too."""

import re

_RULE_RE = re.compile(r"^\|[-| :]+\|$")
_SPLIT_DECIMAL_RE = re.compile(r"\d \.\d")
# A caption cut off mid-sentence ends on a comma, hyphen or function word ("...diabetes among").
_CAPTION_CUT_RE = re.compile(
    r"(,|-|\b(and|or|of|the|with|among|for|in|by|to|a|an|from|on))"
    r"\s*(\((continued|part \d+ of \d+)\))?$",
    re.IGNORECASE,
)
TABLE_CEILING_WORDS = 480


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
        "over_480_words": 0,
    }
    for c in tables:
        cap = caption(c["text"])
        if not cap:
            stats["no_caption"] += 1
        elif _CAPTION_CUT_RE.search(cap):
            stats["caption_cut_mid_sentence"] += 1
        rows = _rows(c["text"])
        if rows:
            width = len(rows[0])
            if any(all(i >= len(r) or not r[i] for r in rows) for i in range(width)):
                stats["all_empty_columns"] += 1
            if len(split_header(clean_rows(rows))[0]) > 1:  # header rows above numeric data
                stats["multi_row_header"] += 1
        if _SPLIT_DECIMAL_RE.search(c["text"]):
            stats["split_decimals"] += 1
        words = [w for w in c["text"].split() if w.strip("|-:")]  # real words, no table markup
        if len(words) > TABLE_CEILING_WORDS:
            stats["over_480_words"] += 1
    stats["continuation_without_caption_or_header"] = sum(
        is_continuation(a, b) for a, b in zip(chunks, chunks[1:], strict=False)
    )
    return stats


# --- table_v2 rendering (corpus v2; corpus v1 keeps Table.to_markdown unchanged) ---

TABLE_FORMATS = ("v1", "v2")
MAX_HEADER_ROWS = 4
MAX_TABLE_WORDS = 450  # body words per v2 table chunk; longer tables split, header repeated
_DECIMAL_RE = re.compile(r"(\d) \.(\d)")
_NUMERIC_CELL_RE = re.compile(r"^[<>\u2264\u2265~+\-\u2212\u2013]?\s*\d")


def rejoin_decimals(text: str) -> str:
    """'26 .8' -> '26.8': PDF text layers sometimes put a space before a decimal point."""
    return _DECIMAL_RE.sub(r"\1.\2", text)


def _pad(rows: list[list[str]]) -> list[list[str]]:
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    rows = _pad(rows)
    keep = [i for i in range(len(rows[0]) if rows else 0) if any(r[i].strip() for r in rows)]
    return [[r[i] for i in keep] for r in rows]


def _is_header_row(row: list[str]) -> bool:
    """A header row has no numeric data: fewer than half of its filled cells after the first
    column start with a number ('(95% CI)' and 'Number in Millions' are header text). A row
    with no filled cell after the first column is a section label, not a header."""
    cells = [c for c in row[1:] if c.strip()]
    if not cells:  # a section label ("Age in years") or a blank row ends the header
        return False
    return sum(bool(_NUMERIC_CELL_RE.match(c.strip())) for c in cells) * 2 < len(cells)


def split_header(rows: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    """Leading header rows and the body. Rows are merged into the header only when a numeric
    data row follows within MAX_HEADER_ROWS rows; a text-only table (or one whose first numeric
    row comes later) keeps its first row as the header, so its text rows are never merged."""
    first_data = next((i for i, r in enumerate(rows) if not _is_header_row(r)), None)
    if first_data is None or first_data == 0 or first_data > MAX_HEADER_ROWS:
        return rows[:1], rows[1:]
    return rows[:first_data], rows[first_data:]


def merge_header(header: list[list[str]]) -> list[str]:
    """One header cell per column: the column's header fragments joined top to bottom, so a
    unit row ('Number in Millions', '(95% CI)') stays next to its column name."""
    header = _pad(header)
    return [" ".join(r[i].strip() for r in header if r[i].strip()) for i in range(len(header[0]))]


def markdown(header: list[str], body: list[list[str]]) -> str:
    rows = _pad([header, *body])
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def full_caption(caption: str, caption_more: str) -> str:
    return " ".join(p for p in (caption.strip(), caption_more.strip()) if p)


def clean_rows(rows: list[list[str]]) -> list[list[str]]:
    return drop_empty_columns([[rejoin_decimals(c) for c in r] for r in rows])


def render_v2(caption: str, header: list[str], body: list[list[str]]) -> list[str]:
    """Chunk texts for one table: caption, one merged header row, body rows. A table whose body
    exceeds MAX_TABLE_WORDS is split into parts of whole rows, each repeating the header."""
    parts: list[list[list[str]]] = [[]]
    words = 0
    for row in body:
        n = len(" ".join(row).split())
        if parts[-1] and words + n > MAX_TABLE_WORDS:
            parts.append([])
            words = 0
        parts[-1].append(row)
        words += n
    texts = []
    for i, part in enumerate(parts, 1):
        label = caption
        if len(parts) > 1:
            label = (
                f"{caption} (part {i} of {len(parts)})"
                if caption
                else f"(part {i} of {len(parts)})"
            )
        table = markdown(header, part)
        texts.append(f"{label}\n\n{table}" if label else table)
    return texts
