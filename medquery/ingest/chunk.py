"""Section-aware chunking: chunks never cross a heading, and each table is its own chunk."""

import re
from collections import Counter
from dataclasses import dataclass

from medquery.ingest import tables
from medquery.ingest.blocks import Page, Table, TextBox
from medquery.ingest.headings import HeadingDetector, is_bold_block

TARGET_WORDS = 300
MAX_WORDS = 450  # packing limit
SHORT_WORDS = 30  # chunks shorter than this merge into a neighbour in the same section
CEILING_WORDS = MAX_WORDS + SHORT_WORDS  # hard upper bound after merging
MIN_WORDS = 5  # a section with less text than this is dropped as noise
ROOT_SHARE = 0.8  # a top-level heading this common is the document title, not a section
# Exercise sections: unanswered questions, some with deliberately false statements.
QUIZ_SECTION_RE = re.compile(
    r"knowledge check|self[- ]?assessment|review questions|\bquiz\b", re.IGNORECASE
)


@dataclass
class Chunk:
    section: str
    page_start: int
    page_end: int
    kind: str  # "text", "table" or "quiz" (exercise items; excluded from retrieval by default)
    source: str  # "text" or "ocr"
    ocr_confidence: float | None
    text: str
    id: str = ""


@dataclass
class _Para:
    text: str
    page: int
    source: str
    confidence: float | None
    last_page: int = 0  # set when a paragraph continues onto a later page

    def __post_init__(self) -> None:
        self.last_page = max(self.last_page, self.page)


SENTENCE_END = (".", "!", "?", ":", ";", '"', "”", ")", "]")
CONTINUATION_MIN_WORDS = 5  # only prose continues; label fragments ("doolB") stay separate


def add_paragraph(paras: list[_Para], para: _Para) -> None:
    """Append `para`, or join it to the previous paragraph when that one is prose that stops
    mid-sentence (a column or page break) and `para` continues in lowercase."""
    if paras:
        prev = paras[-1]
        continues = (
            prev.source == para.source
            and len(prev.text.split()) >= CONTINUATION_MIN_WORDS
            and not prev.text.rstrip().endswith(SENTENCE_END)
            and para.text[:1].islower()
        )
        if continues:
            paras[-1] = _Para(
                join_lines([prev.text, para.text]),
                prev.page,
                prev.source,
                prev.confidence,
                last_page=para.last_page,
            )
            return
    paras.append(para)


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines, rejoining words hyphenated across a line break."""
    out = ""
    for line in lines:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return out


def page_items(page: Page) -> list[TextBox | Table]:
    """Boxes in reading order, with each table placed before the first box below its top."""
    items: list[TextBox | Table] = list(page.boxes)
    for table in sorted(page.tables, key=lambda t: t.top, reverse=True):
        index = next(
            (i for i, it in enumerate(items) if isinstance(it, TextBox) and it.top >= table.top),
            len(items),
        )
        items.insert(index, table)
    return items


def _split_long(para: _Para) -> list[_Para]:
    """Split an over-long paragraph at sentence ends, or at word boundaries as a last resort."""
    words = para.text.split()
    if len(words) <= MAX_WORDS:
        return [para]
    sentences = re.split(r"(?<=[.!?])\s+", para.text)
    if max(len(s.split()) for s in sentences) > MAX_WORDS:
        sentences = [
            " ".join(words[i : i + TARGET_WORDS]) for i in range(0, len(words), TARGET_WORDS)
        ]
    parts, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence]).split()) > TARGET_WORDS:
            parts.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        parts.append(" ".join(current))
    return [_Para(p, para.page, para.source, para.confidence, para.last_page) for p in parts]


def _words(group: list[_Para]) -> int:
    return sum(len(p.text.split()) for p in group)


def merge_short(groups: list[list[_Para]]) -> list[list[_Para]]:
    """Merge each group under SHORT_WORDS into its smaller neighbour.

    Called once per section, so merging never crosses a section boundary.
    """
    groups = list(groups)
    while len(groups) > 1:
        short = next((i for i, g in enumerate(groups) if _words(g) < SHORT_WORDS), None)
        if short is None:
            break
        left = _words(groups[short - 1]) if short > 0 else float("inf")
        right = _words(groups[short + 1]) if short + 1 < len(groups) else float("inf")
        if left <= right:
            groups[short - 1] = groups[short - 1] + groups[short]
        else:
            groups[short + 1] = groups[short] + groups[short + 1]
        del groups[short]
    return groups


def strip_shared_root(chunks: list[Chunk]) -> None:
    """Drop top-level headings shared by most chunks: they are the document title.

    Repeats, because covers often stack several title lines (title, then subtitle).
    """
    while True:
        roots = [c.section.split(" > ")[0] for c in chunks if " > " in c.section]
        if not roots:
            return
        root = max(set(roots), key=roots.count)
        if roots.count(root) < ROOT_SHARE * len(chunks):
            return
        prefix = root + " > "
        for chunk in chunks:
            if chunk.section.startswith(prefix):
                chunk.section = chunk.section[len(prefix) :]


REAL_WORD_RE = re.compile(r"[A-Za-z]{3,}")
DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d+$")
LABEL_MIN_LINES = 6
PROSE_LINE_WORDS = 6  # a line this long is prose, so the chunk is not just labels


def junk_reason(chunk: Chunk) -> str | None:
    """Why a text chunk is junk (page furniture, figure labels, bare numbers), else None."""
    if chunk.kind != "text":
        return None
    lines = [line for line in chunk.text.splitlines() if line.strip()]
    if not REAL_WORD_RE.search(chunk.text):
        return "no real words (bare numbers or references)"
    if sum(1 for line in lines if DOT_LEADER_RE.search(line)) >= 0.5 * len(lines):
        return "contents listing"
    tokens_per_line = sorted(len(line.split()) for line in lines)
    only_labels = tokens_per_line[-1] < PROSE_LINE_WORDS  # recipes, nutrient panels have prose
    if len(lines) >= LABEL_MIN_LINES and tokens_per_line[len(lines) // 2] <= 2 and only_labels:
        return "scattered labels (figure, axis or rotated text)"
    return None


def _in_section(section: str, parent: str) -> bool:
    return section == parent or section.startswith(parent + " > ")


def _labelled(chunk: Chunk, parent: str) -> str:
    """Chunk text, prefixed by its subheading relative to `parent` when it has one."""
    if chunk.section == parent:
        return chunk.text
    return f"{chunk.section[len(parent) + 3 :]}\n{chunk.text}"


def _merge_pair(first: Chunk, second: Chunk, parent: str) -> Chunk:
    confidences = [c.ocr_confidence for c in (first, second) if c.ocr_confidence is not None]
    return Chunk(
        section=parent,
        page_start=min(first.page_start, second.page_start),
        page_end=max(first.page_end, second.page_end),
        kind="text",
        source="ocr" if "ocr" in (first.source, second.source) else "text",
        ocr_confidence=round(sum(confidences) / len(confidences), 1) if confidences else None,
        text=f"{_labelled(first, parent)}\n\n{_labelled(second, parent)}",
    )


def merge_lonely_siblings(chunks: list[Chunk]) -> list[Chunk]:
    """Merge a short chunk that is alone in its section into an adjacent chunk that shares its
    parent section. The result takes the parent's path and keeps each subheading as a first line.

    A top-level section has no parent, so merging never crosses top-level sections.
    """
    chunks = list(chunks)
    while True:
        counts = Counter(c.section for c in chunks if c.kind == "text")
        for i, chunk in enumerate(chunks):
            lonely_short = (
                chunk.kind == "text"
                and len(chunk.text.split()) < SHORT_WORDS
                and counts[chunk.section] == 1
                and " > " in chunk.section
            )
            if not lonely_short:
                continue
            parent = chunk.section.rsplit(" > ", 1)[0]
            options = []
            for j in (i - 1, i + 1):
                neighbour_ok = (
                    0 <= j < len(chunks)
                    and chunks[j].kind == "text"
                    and _in_section(chunks[j].section, parent)
                )
                if not neighbour_ok:
                    continue
                lo, hi = sorted((i, j))
                merged = _merge_pair(chunks[lo], chunks[hi], parent)
                if len(merged.text.split()) <= CEILING_WORDS:
                    options.append((len(chunks[j].text.split()), lo, merged))
            if options:
                _, lo, merged = min(options, key=lambda option: (option[0], option[1]))
                chunks[lo : lo + 2] = [merged]
                break
        else:
            return chunks


def _pack(paras: list[_Para], section: str) -> list[Chunk]:
    pieces = [piece for para in paras for piece in _split_long(para)]
    groups: list[list[_Para]] = []
    current: list[_Para] = []
    count = 0
    for piece in pieces:
        words = len(piece.text.split())
        if current and (count + words > MAX_WORDS or count >= TARGET_WORDS):
            groups.append(current)
            current, count = [], 0
        current.append(piece)
        count += words
    if current:
        groups.append(current)

    chunks = []
    for group in merge_short(groups):
        text = "\n\n".join(p.text for p in group)
        if len(text.split()) < MIN_WORDS:
            continue
        confidences = [p.confidence for p in group if p.confidence is not None]
        chunks.append(
            Chunk(
                section=section,
                page_start=min(p.page for p in group),
                page_end=max(p.last_page for p in group),
                kind="text",
                source="ocr" if any(p.source == "ocr" for p in group) else "text",
                ocr_confidence=round(sum(confidences) / len(confidences), 1)
                if confidences
                else None,
                text=text,
            )
        )
    return chunks


def mark_quiz(chunks: list[Chunk]) -> None:
    """Label chunks in exercise sections (e.g. "Knowledge Check") as kind="quiz"."""
    for chunk in chunks:
        if QUIZ_SECTION_RE.search(chunk.section):
            chunk.kind = "quiz"


def _table_v2(item: Table, page: int, last: tuple | None) -> tuple[list[str], tuple]:
    """Corpus v2 table chunks. A captionless table on the page after a table with the same
    raw width continues it: it gets '<caption> (continued)' and that table's merged header."""
    width = max(len(r) for r in item.rows)
    caption = tables.full_caption(item.caption, item.caption_more)
    rows = tables.clean_rows(item.rows)
    continued = not caption and last is not None and last[0] == page - 1 and last[3] == width
    if continued and len(rows[0]) == len(last[2]):
        caption = f"{last[1]} (continued)" if last[1] else "(continued)"
        header, body = last[2], rows
    else:
        head, body = tables.split_header(rows)
        header = tables.merge_header(head)
        if continued:  # same raw width but different columns after cleaning: keep its own
            caption = f"{last[1]} (continued)" if last[1] else "(continued)"
    base_caption = caption.removesuffix(" (continued)") if continued else caption
    return tables.render_v2(caption, header, body), (page, base_caption, header, width)


def chunk_document(
    doc_id: str,
    pages: list[Page],
    dropped: list[tuple[str, Chunk]] | None = None,
    table_format: str = "v1",
) -> list[Chunk]:
    """Chunk one document. Junk chunks are removed and appended to `dropped` with a reason.

    Order matters: quiz chunks are labelled before junk filtering and sibling merging, so they
    are never dropped and never merged into (or with) real content.
    """
    detector = HeadingDetector(pages)
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # (level, heading text)
    paras: list[_Para] = []
    heading_open = False  # True right after a heading, before any body text
    last_table: tuple | None = None  # v2: (page, caption, merged header, raw width)

    def section() -> str:
        return " > ".join(text for _, text in stack)

    def flush() -> None:
        chunks.extend(_pack(paras, section()))
        paras.clear()

    for page in pages:
        conf = page.ocr_confidence if page.source == "ocr" else None
        for item in page_items(page):
            if isinstance(item, Table):
                flush()
                heading_open = False
                if table_format == "v2":
                    texts, last_table = _table_v2(item, page.number, last_table)
                else:
                    texts = [
                        f"{item.caption}\n\n{item.to_markdown()}"
                        if item.caption
                        else item.to_markdown()
                    ]
                for text in texts:
                    chunks.append(
                        Chunk(section(), page.number, page.number, "table", page.source, conf, text)
                    )
                continue
            body: list[str] = []
            bold_block = is_bold_block(item)
            for line in item.lines:
                level = detector.level(line, bold_block) if page.source == "text" else None
                if level is None:
                    body.append(line.text)
                    heading_open = False
                    continue
                if body:
                    add_paragraph(paras, _Para(join_lines(body), page.number, page.source, conf))
                    body = []
                if heading_open and stack and stack[-1][0] == level:
                    stack[-1] = (level, f"{stack[-1][1]} {line.text}")  # multi-line heading
                    continue
                flush()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, line.text))
                heading_open = True
            if body:
                add_paragraph(paras, _Para(join_lines(body), page.number, page.source, conf))
    flush()

    strip_shared_root(chunks)
    mark_quiz(chunks)
    kept = []
    for chunk in chunks:
        reason = junk_reason(chunk)
        if reason is None:
            kept.append(chunk)
        elif dropped is not None:
            dropped.append((reason, chunk))
    chunks = merge_lonely_siblings(kept)
    for n, chunk in enumerate(chunks):
        chunk.id = f"{doc_id}:{n:04d}"
    return chunks
