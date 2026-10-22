"""Okapi BM25 over the chunk index texts (in memory; about 1,100 chunks, so no search engine).

Standard parameters k1 = 1.2, b = 0.75 (Robertson et al.), not tuned. Tokens keep decimals and
slashes ("130/80", "2.5") because clinical questions hinge on them; no stemming.
"""

import math
import re
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./][a-z0-9]+)*")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "them",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.lengths = np.array([sum(c.values()) for c in self.tf], dtype=np.float64)
        self.avgdl = float(self.lengths.mean()) if len(docs) else 0.0
        df = Counter(term for counts in self.tf for term in counts)
        n = len(docs)
        # Lucene's non-negative IDF variant: log(1 + (N - df + 0.5) / (df + 0.5)).
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                f = counts.get(term)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """(document index, score) for the k best documents with a positive score.
        Ties keep index order, so results are deterministic."""
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
