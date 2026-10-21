"""Data types passed between PDF extraction, OCR, heading detection and chunking."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Line:
    text: str
    size: float  # median font size in points; 0 for OCR text
    bold: bool


@dataclass
class TextBox:
    """A block of lines in reading order (a pdfminer text box, or an OCR paragraph)."""

    lines: list[Line]
    top: float  # vertical position as a fraction of page height
    bottom: float


@dataclass(frozen=True)
class Table:
    rows: list[list[str]]
    caption: str
    top: float  # fraction of page height

    def to_markdown(self) -> str:
        width = max(len(row) for row in self.rows)
        rows = [row + [""] * (width - len(row)) for row in self.rows]
        out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
        out += ["| " + " | ".join(row) + " |" for row in rows[1:]]
        return "\n".join(out)


@dataclass
class Page:
    number: int  # 1-based physical page index
    boxes: list[TextBox] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    source: str = "text"  # "text" or "ocr"
    ocr_confidence: float | None = None  # mean word confidence (0-100) for OCR pages
    ocr_words: int = 0
