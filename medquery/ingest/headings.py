"""Heading detection from font size and weight, relative to the document's body text."""

import re
from collections import Counter

from medquery.ingest.blocks import Line, Page, TextBox

SIZE_RATIO = 1.15  # a line this much larger than body text is a heading
MAX_HEADING_WORDS = 12
MAX_BOLD_BLOCK_LINES = 3
NOT_HEADING_RE = re.compile(
    r"^[•●○■▪◦✓*\-–—]"  # bullet items
    r"|https?://|www\.|@"  # links and emails
    r"|\d{3}[-.\s]\d{3,4}"  # phone numbers
)


def _bucket(size: float) -> float:
    return round(size * 2) / 2


def body_size(pages: list[Page]) -> float:
    """The most common font size, weighted by characters."""
    counts: Counter[float] = Counter()
    for page in pages:
        for box in page.boxes:
            for line in box.lines:
                if line.size:
                    counts[_bucket(line.size)] += len(line.text)
    return counts.most_common(1)[0][0] if counts else 0.0


def heading_text_ok(text: str) -> bool:
    words = text.split()
    return (
        0 < len(words) <= MAX_HEADING_WORDS
        and any(ch.isalpha() for ch in text)
        and not text.endswith((".", ",", ";"))
        and not NOT_HEADING_RE.search(text)
    )


def is_bold_block(box: TextBox) -> bool:
    """A short block made only of bold lines: the shape of a bold run-on heading."""
    return len(box.lines) <= MAX_BOLD_BLOCK_LINES and all(line.bold for line in box.lines)


class HeadingDetector:
    """Assigns heading levels: larger sizes rank higher; bold body-size lines rank lowest."""

    def __init__(self, pages: list[Page]):
        self.body = body_size(pages)
        sizes = {
            _bucket(line.size)
            for page in pages
            for box in page.boxes
            for line in box.lines
            if self._large(line) and heading_text_ok(line.text)
        }
        self.levels = {size: i + 1 for i, size in enumerate(sorted(sizes, reverse=True))}
        self.bold_level = len(self.levels) + 1

    def _large(self, line: Line) -> bool:
        return bool(self.body and line.size) and _bucket(line.size) >= self.body * SIZE_RATIO

    def level(self, line: Line, bold_block: bool) -> int | None:
        """Heading level of `line`, or None for body text.

        A large line is a heading by size. A body-size bold line is a heading only when
        `bold_block` is True: its whole text box is a short all-bold block (see is_bold_block),
        so bold labels and bold words inside paragraphs stay body text.
        """
        if not line.size or not heading_text_ok(line.text):
            return None
        if self._large(line):
            return self.levels.get(_bucket(line.size))
        if line.bold and bold_block and _bucket(line.size) >= self.body * 0.95:
            return self.bold_level
        return None
