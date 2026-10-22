"""Reciprocal rank fusion (Cormack, Clarke & Buttcher, 2009)."""

RRF_K = 60  # the paper's constant; RRF is insensitive to it, so it is not tuned


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank of d), ranks from 1.

    Ties break by the best single rank, then by the id, so the order is deterministic."""
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            best[doc_id] = min(best.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best[item[0]], item[0]))
