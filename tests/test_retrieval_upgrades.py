"""Phase 3 retrieval: BM25, reciprocal rank fusion, the reranker wrapper and the hybrid
pipeline. All offline: fake embedder and fake cross-encoder."""

import math
from dataclasses import asdict

import pytest

from medquery.eval import runner
from medquery.eval.runner import CONFIGS, RETRIEVAL_DEFAULTS, RunConfig, build_retriever, run_id
from medquery.retrieval.bm25 import BM25, tokenize
from medquery.retrieval.fusion import RRF_K, rrf
from medquery.retrieval.pipeline import HybridRetriever, index_text
from medquery.retrieval.rerank import FastEmbedReranker, rerank
from tests.test_rag import CHUNKS, FakeEmbedder, chunk

# --- BM25 ---


def test_bm25_matches_a_hand_computed_score():
    bm25 = BM25(["aa bb", "aa cc cc", "dd"])
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))  # "cc" is in 1 of 3 documents
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / 2)  # document length 3, average length 2
    expected = idf * 2 * (1.2 + 1) / (2 + norm)  # term frequency 2
    scores = bm25.scores("cc")
    assert scores[1] == pytest.approx(expected)
    assert scores[0] == 0 and scores[2] == 0


def test_bm25_weighs_rare_terms_more():
    bm25 = BM25(["common rare", "common", "common", "common"])
    assert bm25.idf["rare"] > bm25.idf["common"]
    assert bm25.top("common rare", 4)[0][0] == 0


def test_tokens_keep_readings_and_decimals_and_drop_stopwords():
    assert tokenize("Is the target 130/80 or 2.5 mg?") == ["target", "130/80", "2.5", "mg"]


def test_bm25_empty_or_unknown_query_returns_nothing():
    bm25 = BM25(["aa bb", "cc"])
    assert bm25.top("", 5) == []
    assert bm25.top("zz the", 5) == []


# --- reciprocal rank fusion ---


def test_rrf_formula_and_items_found_by_one_list():
    fused = dict(rrf([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)
    assert [d for d, _ in rrf([["a", "b"], ["b", "c"]])][0] == "b"
    assert RRF_K == 60


def test_rrf_ties_break_by_best_rank_then_id():
    # "x" and "y" both score 1/61 + 1/62; "x" has no better rank, so the id decides.
    assert [d for d, _ in rrf([["y", "x"], ["x", "y"]])] == ["x", "y"]
    # With k = 0, "z" (rank 1 once) and "a" (rank 2 twice) both score 1; "z" has the better
    # single rank, so it comes first although its id sorts later.
    fused = rrf([["z"], ["b", "a"], ["c", "a"]], k=0)
    assert dict(fused)["z"] == pytest.approx(dict(fused)["a"])
    order = [d for d, _ in fused]
    assert order.index("z") < order.index("a")


# --- reranker wrapper ---


class FakeEncoder:
    """Scores a passage by how often it says 'sodium'; long texts count as truncated."""

    model_name = "fake/cross"

    def __init__(self, max_chars: int = 1000):
        self.seen: list[list[str]] = []
        self.max_chars = max_chars

    def score(self, query, texts):
        self.seen.append(list(texts))
        return [float(t.lower().split().count("sodium")) for t in texts]

    def count_truncated(self, query, texts):
        return sum(len(query) + len(t) > self.max_chars for t in texts)


def test_rerank_reorders_by_score_and_only_scores_the_top_n():
    candidates = [{"id": "a", "t": "x"}, {"id": "b", "t": "sodium"}, {"id": "c", "t": "sodium"}]
    encoder = FakeEncoder()
    out = rerank("q", candidates, lambda c: c["t"], encoder, n=2)
    assert [c["id"] for c, _ in out] == ["b", "a"]  # "c" is beyond n: never scored
    assert encoder.seen == [["x", "sodium"]]


def test_rerank_is_stable_on_equal_scores():
    candidates = [{"id": str(n), "t": "same"} for n in range(5)]
    out = rerank("q", candidates, lambda c: c["t"], FakeEncoder(), n=5)
    assert [c["id"] for c, _ in out] == ["0", "1", "2", "3", "4"]


def test_unknown_reranker_is_refused_before_any_download():
    with pytest.raises(ValueError, match="licence"):
        FastEmbedReranker("someone/unlicensed-reranker")


# --- pipeline ---


def retriever(tmp_path, **kw):
    return HybridRetriever(FakeEmbedder(), CHUNKS, embed_cache=tmp_path / "e.npz", **kw)


def test_hybrid_returns_only_the_given_chunks(tmp_path):
    hits, seconds = retriever(tmp_path, use_bm25=True).retrieve("sodium blood pressure", 3)
    assert hits[0].chunk["id"] == "htn:0001" and seconds >= 0
    assert [h.rank for h in hits] == [1, 2, 3]
    assert {h.chunk["id"] for h in hits} <= {c["id"] for c in CHUNKS}


def test_quiz_chunks_never_reach_any_mode(tmp_path):
    quiz = chunk("tb:0099", "sodium sodium sodium sodium", kind="quiz")
    for kw in ({"use_bm25": True}, {"use_bm25": True, "reranker": FakeEncoder()}):
        r = HybridRetriever(
            FakeEmbedder(),
            [c for c in [*CHUNKS, quiz] if c["kind"] != "quiz"],  # as load_chunks() gives them
            embed_cache=tmp_path / "e.npz",
            **kw,
        )
        assert "tb:0099" not in {h.chunk["id"] for h in r.retrieve("sodium", 4)[0]}


def test_headers_change_the_index_text_but_not_what_the_generator_sees(tmp_path):
    encoder = FakeEncoder()
    r = retriever(tmp_path, use_bm25=True, headers=True, reranker=encoder)
    hits, _ = r.retrieve("sodium", 3)
    original = {c["id"]: c["text"] for c in CHUNKS}
    assert all(h.chunk["text"] == original[h.chunk["id"]] for h in hits)
    assert all(t.startswith("Doc | Treatment\n") for t in encoder.seen[0])  # reranker input
    assert r.texts["tb:0001"] == "Doc | Treatment\nrifampin dose daily rifampin"
    assert index_text(CHUNKS[0], headers=False) == CHUNKS[0]["text"]


def test_rerank_records_how_many_inputs_were_truncated(tmp_path):
    r = retriever(tmp_path, use_bm25=True, reranker=FakeEncoder(max_chars=25), rerank_n=2)
    r.retrieve("sodium", 3)
    assert r.last_stats["rerank_pairs"] == 2
    assert 0 <= r.last_stats["rerank_truncated"] <= 2
    long = retriever(tmp_path, use_bm25=True, reranker=FakeEncoder(max_chars=0), rerank_n=3)
    long.retrieve("sodium", 3)
    assert long.last_stats == {"rerank_pairs": 3, "rerank_truncated": 3}


def test_without_a_reranker_there_are_no_rerank_stats(tmp_path):
    r = retriever(tmp_path, use_bm25=True)
    r.retrieve("sodium", 2)
    assert r.last_stats == {}


# --- configs and run ids ---


def test_baseline_run_id_is_unchanged_by_the_new_fields():
    """The Phase 2 run id hashed only the Phase 2 fields; new fields at their defaults are
    left out, so the baseline resumes the same run directory."""
    phase2 = {k: v for k, v in asdict(RunConfig()).items() if k not in RETRIEVAL_DEFAULTS}
    import hashlib
    import json

    payload = {**phase2, "split": "dev", "golden": "g", "chunks": "c"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    assert run_id(CONFIGS["baseline"], "dev", "g", "c") == expected
    ids = {run_id(cfg, "dev", "g", "c") for cfg in CONFIGS.values()}
    assert len(ids) == len(CONFIGS)  # every config gets its own run


def test_configs_are_cumulative():
    assert CONFIGS["hybrid"].retriever == "hybrid" and not CONFIGS["hybrid"].rerank_model
    assert CONFIGS["hybrid_rerank"].rerank_model == "BAAI/bge-reranker-base"
    assert CONFIGS["hybrid_rerank"].rerank_n == 20 and not CONFIGS["hybrid_rerank"].headers
    assert CONFIGS["hybrid_rerank_headers"].headers
    for name in ("hybrid_k8", "hybrid_rerank_k8", "hybrid_rerank_headers_k8"):
        base = CONFIGS[name.removesuffix("_k8")]
        assert CONFIGS[name].k == 8 and asdict(CONFIGS[name]) | {"k": 5, "name": base.name} == (
            asdict(base)
        )
    assert {c.generator for c in CONFIGS.values()} == {RunConfig().generator}
    assert {c.judge_prompt for c in CONFIGS.values()} == {RunConfig().judge_prompt}


def test_build_retriever_picks_the_pipeline(monkeypatch):
    built = []
    monkeypatch.setattr(runner, "Retriever", lambda e, c: built.append("dense") or "dense")
    monkeypatch.setattr(runner, "HybridRetriever", lambda e, c, **kw: built.append(kw) or "hybrid")
    assert build_retriever(CONFIGS["baseline"], CHUNKS, FakeEmbedder()) == "dense"
    encoder = FakeEncoder()
    assert build_retriever(CONFIGS["hybrid_rerank_headers"], CHUNKS, FakeEmbedder(), encoder)
    kw = built[-1]
    assert kw["use_bm25"] and kw["headers"] and kw["reranker"] is encoder and kw["rerank_n"] == 20
    build_retriever(CONFIGS["hybrid"], CHUNKS, FakeEmbedder())
    assert built[-1]["reranker"] is None and not built[-1]["headers"]
