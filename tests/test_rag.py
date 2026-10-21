import numpy as np
import pytest

from medquery.generation.answer import answer
from medquery.generation.prompts import (
    ABSTAIN_TEXT,
    ANSWER_SYSTEM,
    answer_messages,
    extract_citations,
    is_abstention,
)
from medquery.llm.base import Completion
from medquery.retrieval.dense import DenseIndex, Hit, Retriever, load_chunks
from medquery.retrieval.embed import embed_cached, text_key

VOCAB = ["rifampin", "isoniazid", "insulin", "sodium", "blood", "pressure", "dose", "diabetes"]


class FakeEmbedder:
    """Bag-of-words vectors over a tiny vocabulary: deterministic and offline."""

    model_name = "fake/bow"

    def __init__(self):
        self.passage_calls = 0

    def _vec(self, text: str) -> np.ndarray:
        words = text.lower().split()
        return np.array([words.count(w) + 0.01 for w in VOCAB], dtype=np.float32)

    def embed_passages(self, texts):
        self.passage_calls += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def chunk(chunk_id: str, text: str, kind: str = "text", superseded: str = "") -> dict:
    return {
        "id": chunk_id,
        "doc_id": chunk_id.split(":")[0],
        "title": "Doc",
        "publisher": "CDC",
        "as_of": "2003" if superseded else "2020",
        "superseded": superseded,
        "section": "Treatment",
        "page_start": 8,
        "page_end": 8,
        "kind": kind,
        "text": text,
    }


CHUNKS = [
    chunk("tb:0001", "rifampin dose daily rifampin"),
    chunk("dm:0001", "insulin diabetes insulin"),
    chunk("htn:0001", "sodium blood pressure sodium", superseded="Superseded by 2017 ACC/AHA."),
]


def test_dense_search_ranks_by_cosine_similarity():
    embedder = FakeEmbedder()
    index = DenseIndex(CHUNKS, embedder.embed_passages([c["text"] for c in CHUNKS]))
    hits = index.search(embedder.embed_query("insulin diabetes"), k=3)
    assert hits[0].chunk["id"] == "dm:0001"
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score > hits[1].score >= hits[2].score


def test_load_chunks_excludes_quiz_unless_requested(tmp_path):
    import json

    path = tmp_path / "chunks.jsonl"
    rows = [chunk("a:0001", "x"), chunk("a:0002", "quiz question", kind="quiz")]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert [c["id"] for c in load_chunks(path)] == ["a:0001"]
    assert [c["id"] for c in load_chunks(path, include_quiz=True)] == ["a:0001", "a:0002"]


def test_embedding_cache_reuses_vectors_by_text_hash(tmp_path):
    embedder = FakeEmbedder()
    path = tmp_path / "emb.npz"
    texts = [c["text"] for c in CHUNKS]
    first, computed = embed_cached(texts, embedder, path)
    assert computed == 3
    second, computed = embed_cached(texts + ["insulin dose"], embedder, path)
    assert computed == 1  # only the new text
    assert embedder.passage_calls == 4
    np.testing.assert_allclose(first, second[:3])
    with np.load(path) as data:
        assert text_key("insulin dose") in data["keys"].tolist()


def test_prompt_lists_passages_with_ids_and_superseded_marker():
    hits = [Hit(c, 0.9, i) for i, c in enumerate(CHUNKS, 1)]
    system, user = answer_messages("What is the rifampin dose?", hits)
    assert system["content"] == ANSWER_SYSTEM
    assert "[tb:0001] Doc | CDC | as of 2020 | p.8 | Treatment" in user["content"]
    assert "[htn:0001] Doc | CDC | as of 2003 | SUPERSEDED | p.8" in user["content"]
    assert user["content"].endswith("Question: What is the rifampin dose?")
    assert ABSTAIN_TEXT in system["content"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (ABSTAIN_TEXT, True),
        ("i cannot answer this from the provided documents", True),
        ("I cannot answer this from the provided documents. The passages cover only adults.", True),
        ("Rifampin is given daily [tb:0001].", False),
        ("The documents say I cannot answer this from the provided documents.", False),
    ],
)
def test_abstention_detection(text, expected):
    assert is_abstention(text) is expected


def test_citation_extraction_handles_lists_and_duplicates():
    text = "Dose A [tb-ltbi-2020:0020]. Dose B [tb-ltbi-2020:0020; dm-stats-2020:0003, x:0001]."
    assert extract_citations(text) == ["tb-ltbi-2020:0020", "dm-stats-2020:0003", "x:0001"]
    assert extract_citations("no citations [see Table 4]") == []


class FakeLLM:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return Completion(text=self.text, model=kwargs["model"], finish_reason="stop")


def test_answer_retrieves_prompts_and_parses_citations(tmp_path):
    retriever = Retriever(FakeEmbedder(), CHUNKS, embed_cache=tmp_path / "emb.npz")
    llm = FakeLLM("Rifampin is taken daily [tb:0001]. See also [zz:9999].")
    result = answer("rifampin dose", retriever, llm, k=2, model="gen")
    assert len(result.hits) == 2 and result.hits[0].chunk["id"] == "tb:0001"
    assert result.cited_ids == ["tb:0001", "zz:9999"]
    assert result.invalid_citations == ["zz:9999"]
    assert not result.abstained
    call = llm.calls[0]
    assert call["model"] == "gen" and call["temperature"] == 0.0 and call["seed"] == 0
    assert call["reasoning_effort"] == "low"


def test_answer_marks_abstention(tmp_path):
    retriever = Retriever(FakeEmbedder(), CHUNKS, embed_cache=tmp_path / "emb.npz")
    result = answer("x", retriever, FakeLLM(ABSTAIN_TEXT), k=1, model="gen")
    assert result.abstained
    assert result.cited_ids == []
