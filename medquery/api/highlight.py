"""Which part of a cited passage a sentence most likely came from, for highlighting. The answer
prompt cites passages, not spans, so this is a best guess by content-word overlap."""

import re

from medquery.generation.prompts import extract_citations
from medquery.retrieval.bm25 import tokenize

MIN_OVERLAP = 0.3  # share of the sentence's content words a segment must contain
MAX_SPANS = 2
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> set[str]:
    return {w for w in tokenize(text) if len(w) >= 3 or any(c.isdigit() for c in w)}


def segments(passage: str) -> list[tuple[int, int]]:
    """[start, end) of each line's sentences (a table row is one line, so one segment)."""
    out, offset = [], 0
    for line in passage.split("\n"):
        start = offset
        for piece in _SENTENCE_END_RE.split(line):
            begin = passage.index(piece, start) if piece else start
            if piece.strip():
                out.append((begin, begin + len(piece)))
            start = begin + len(piece)
        offset += len(line) + 1
    return out


def highlights(sentence: str, passage: str) -> list[list[int]]:
    """Spans [start, end) of the passage segments that best match the sentence."""
    cited = extract_citations(sentence)
    target = _words(sentence) - {w for c in cited for w in tokenize(c)}
    if not target:
        return []
    scored = []
    for start, end in segments(passage):
        overlap = len(target & _words(passage[start:end])) / len(target)
        if overlap >= MIN_OVERLAP:
            scored.append((overlap, start, end))
    best = sorted(scored, key=lambda s: -s[0])[:MAX_SPANS]
    return [[start, end] for _, start, end in sorted(best, key=lambda s: s[1])]
