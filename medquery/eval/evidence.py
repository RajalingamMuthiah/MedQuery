"""Map gold evidence, anchored as (doc_id, page, quote), to chunks at evaluation time.

Gold evidence never stores chunk ids, because chunking may change in later phases. Each
quote is located in the chunks of its document whose page range covers its page.
"""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

FUZZY_COVERAGE = 0.8  # share of the quote that matched chunks must cover together
MIN_PIECE_CHARS = 15  # a matching piece shorter than this is ignored

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-"})


def normalise(text: str) -> str:
    """Compare text the same way on both sides: ligatures, quotes/dashes, case, hyphenation,
    table pipes and whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    text = re.sub(r"\|?-{3,}\|?", " ", text)  # Markdown table rules
    text = text.replace("|", " ")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # words hyphenated across a line break
    text = re.sub(r"(\d)\s+(?=[a-zµ%])", r"\1", text)  # "130/80 mmhg" == "130/80mmhg" (PDF kerning)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    page: int
    quote: str


@dataclass(frozen=True)
class MappedEvidence:
    evidence: Evidence
    chunk_ids: tuple[str, ...]
    status: str  # "exact", "fuzzy" or "missing"


class EvidenceMapper:
    def __init__(self, chunks: list[dict]):
        self._by_doc: dict[str, list[tuple[dict, str]]] = defaultdict(list)
        for chunk in chunks:
            self._by_doc[chunk["doc_id"]].append((chunk, normalise(chunk["text"])))

    def map(self, evidence: Evidence) -> MappedEvidence:
        quote = normalise(evidence.quote)
        candidates = [
            (chunk, text)
            for chunk, text in self._by_doc.get(evidence.doc_id, [])
            if chunk["page_start"] <= evidence.page <= chunk["page_end"]
        ]
        exact = tuple(chunk["id"] for chunk, text in candidates if quote and quote in text)
        if exact:
            return MappedEvidence(evidence, exact, "exact")
        return self._fuzzy(evidence, quote, candidates)

    def _fuzzy(self, evidence: Evidence, quote: str, candidates) -> MappedEvidence:
        """A quote split across chunks: each chunk matches a long piece; together they cover it."""
        covered = [False] * len(quote)
        matched: list[str] = []
        for chunk, text in candidates:
            match = SequenceMatcher(None, quote, text, autojunk=False).find_longest_match(
                0, len(quote), 0, len(text)
            )
            if match.size >= MIN_PIECE_CHARS:
                matched.append(chunk["id"])
                for i in range(match.a, match.a + match.size):
                    covered[i] = True
        if quote and matched and sum(covered) >= FUZZY_COVERAGE * len(quote):
            return MappedEvidence(evidence, tuple(matched), "fuzzy")
        return MappedEvidence(evidence, (), "missing")


def coverage(mapped: list[MappedEvidence]) -> dict[str, int]:
    """Counts by status, for the mapping-coverage line in every report."""
    counts = {"exact": 0, "fuzzy": 0, "missing": 0}
    for m in mapped:
        counts[m.status] += 1
    return counts
