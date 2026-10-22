"""Top-up 2: stricter unanswerable checks and the byte-preserving review-CSV append."""

import json

import numpy as np
import pytest

from medquery.eval.golden import REVIEW_COLUMNS, append_review_csv, read_review_csv
from medquery.eval.golden_gen import (
    TOPUPS,
    UNANSWERABLE_TEMPLATE_V2,
    VERIFY_V2_BATCH_TOKENS,
    KeywordScan,
    SourcePage,
    Task,
    batch_passages,
    generation_messages,
    term_in_corpus,
    v2_verifier,
)
from medquery.ingest.manifest import Document
from medquery.llm.base import Completion
from medquery.retrieval.dense import Retriever

FILLER = " ".join(["Isoniazid and rifampin are used to treat latent tuberculosis infection."] * 20)
CHUNKS = [
    {
        "id": f"tb:{n:04d}",
        "doc_id": "tb",
        "page_start": n,
        "page_end": n,
        "kind": "text",
        "text": f"Passage {n}. {FILLER}",
    }
    for n in range(40)
] + [
    {
        "id": "tb:9999",
        "doc_id": "tb",
        "page_start": 1,
        "page_end": 1,
        "kind": "quiz",
        "text": "Quiz: rifapentine monotherapy is the preferred regimen. True or false?",
    }
]


class FakeEmbedder:
    model_name = "fake"

    def embed_passages(self, texts):
        return np.stack([np.array([1.0, len(t) % 7 + 1.0]) for t in texts]).astype(np.float32)

    def embed_query(self, text):
        return np.array([1.0, 1.0], dtype=np.float32)


class ScriptedJudge:
    """Premise calls return `premises`; batch calls return `batch_replies` in order (the last
    reply repeats)."""

    def __init__(self, premises, batch_replies):
        self.premises = premises
        self.batch_replies = batch_replies
        self.batch_calls: list[str] = []

    def complete(self, messages, **kw):
        user = messages[-1]["content"]
        if "List every presupposition" in user:
            reply = {"presuppositions": self.premises}
        else:
            self.batch_calls.append(user)
            reply = self.batch_replies[min(len(self.batch_calls), len(self.batch_replies)) - 1]
        return Completion(text=json.dumps(reply), model=kw["model"], finish_reason="stop")


def verifier(judge, tmp_path):
    retriever = Retriever(
        FakeEmbedder(), [c for c in CHUNKS if c["kind"] != "quiz"], embed_cache=tmp_path / "e.npz"
    )
    scan = KeywordScan([c for c in CHUNKS if c["kind"] != "quiz"])
    return v2_verifier(retriever, scan, judge, CHUNKS)


PREMISES = [
    {"statement": "Isoniazid is used to treat LTBI", "key_terms": ["isoniazid"]},
    {"statement": "Rifampin is used to treat LTBI", "key_terms": ["rifampin"]},
]
CLEAN = {
    "answer_found": False,
    "partial": False,
    "presuppositions": [{"number": 1, "status": "supported"}, {"number": 2, "status": "supported"}],
}
QUESTION = "What is the isoniazid dose adjustment for patients on dialysis taking rifampin?"


def test_accepts_only_when_no_answer_and_every_premise_is_supported(tmp_path):
    judge = ScriptedJudge(PREMISES, [CLEAN])
    verdict, completions, reason = verifier(judge, tmp_path)(QUESTION)
    assert reason is None
    assert len(set(verdict["checked_ids"])) == len(verdict["checked_ids"]) == 30  # 15 + 15, deduped
    assert len(judge.batch_calls) == len(verdict["batches"]) >= 2  # 30 passages need batching
    assert len(completions) == 1 + len(judge.batch_calls)
    assert verdict["version"].startswith("unanswerable_check_v2:")


def test_false_premise_term_missing_from_corpus_rejects_before_any_passage_check(tmp_path):
    premises = [
        {"statement": "Rifapentine monotherapy exists", "key_terms": ["rifapentine monotherapy"]}
    ]
    judge = ScriptedJudge(premises, [CLEAN])
    verdict, _, reason = verifier(judge, tmp_path)(QUESTION)
    assert reason == "false premise: 'rifapentine monotherapy' is not in the corpus"
    assert judge.batch_calls == []  # quiz text, which mentions it, does not count as corpus
    assert verdict["missing_terms"] == ["rifapentine monotherapy"]


def test_answer_in_any_batch_rejects_and_stops_early(tmp_path):
    found = {"answer_found": False, "partial": True, "presuppositions": []}
    judge = ScriptedJudge(PREMISES, [found, CLEAN])
    _, _, reason = verifier(judge, tmp_path)(QUESTION)
    assert reason == "answer found in corpus by verification check (v2)"
    assert len(judge.batch_calls) == 1


def test_contradicted_premise_rejects(tmp_path):
    contradicted = {
        "answer_found": False,
        "partial": False,
        "presuppositions": [{"number": 2, "status": "contradicted"}],
    }
    judge = ScriptedJudge(PREMISES, [contradicted])
    _, _, reason = verifier(judge, tmp_path)(QUESTION)
    assert reason == "false premise: presupposition 2 is contradicted by the corpus"


def test_premise_never_supported_in_any_batch_rejects(tmp_path):
    only_first = {
        "answer_found": False,
        "partial": False,
        "presuppositions": [
            {"number": 1, "status": "supported"},
            {"number": 2, "status": "not_addressed"},
        ],
    }
    judge = ScriptedJudge(PREMISES, [only_first])
    _, _, reason = verifier(judge, tmp_path)(QUESTION)
    assert reason == (
        "premise not established: 'Rifampin is used to treat LTBI' is not supported by any passage"
    )


def test_batches_stay_within_the_token_budget():
    batches = batch_passages(CHUNKS[:30])
    assert sum(len(b) for b in batches) == 30
    for batch in batches:
        assert sum(len(c["text"]) for c in batch) / 3.5 <= VERIFY_V2_BATCH_TOKENS


def test_term_in_corpus_matches_whole_words_only():
    corpus = "isoniazid and inhalation therapy; 3hp regimen"
    assert term_in_corpus("Isoniazid", corpus)
    assert term_in_corpus("3HP", corpus)
    assert not term_in_corpus("INH", corpus)  # must not match inside "inhalation"


def test_topup2_configuration_and_versioned_template():
    groups, _ = TOPUPS["topup2"]
    [group] = groups
    assert (group.type, group.topic, group.target, group.verifier) == (
        "unanswerable",
        "tuberculosis",
        6,
        "v2",
    )
    assert set(group.doc_ids) == {"tb-core-curriculum", "tb-ltbi-2020", "tb-qa-2021"}
    assert group.prompt_id.startswith("golden_gen_v1+unanswerable_v2:")
    doc = Document(
        "tb-qa-2021",
        "TB Q&A",
        "CDC",
        2021,
        "tuberculosis",
        "https://x/a.pdf",
        "Public domain",
        "e",
        "2021",
    )
    page = SourcePage("tb-qa-2021", 9, FILLER, "text", None, 0)
    task = Task("unanswerable", "tuberculosis", [page], 1, template=UNANSWERABLE_TEMPLATE_V2)
    prompt = generation_messages(task, {"tb-qa-2021": doc})[-1]["content"]
    assert "no false premises" in prompt


def write_csv(path, text: str) -> bytes:
    data = "﻿".encode() + text.encode("utf-8")
    path.write_bytes(data)
    return data


def row(row_id: str) -> dict:
    return {c: "" for c in REVIEW_COLUMNS} | {"id": row_id, "question": f"Q {row_id}?"}


def test_append_keeps_existing_bytes_exactly_and_matches_line_endings(tmp_path):
    path = tmp_path / "review.csv"
    header = ",".join(REVIEW_COLUMNS)
    # Hand-edited quirks that a rewrite would normalise away: extra quoting, an en dash.
    existing = (
        f"{header}\r\n"
        '"g001","drop","","Q one?","A, with comma","","factual","tuberculosis","yes","","",'
        '"Note: 2017–2018 range"\r\n'
    )
    old = write_csv(path, existing)
    append_review_csv(path, [row("g097"), row("g098")])
    new = path.read_bytes()
    assert new.startswith(old)
    assert new[len(old) :].count(b"\r\n") == 2 and b"\n" not in new[len(old) :].replace(
        b"\r\n", b""
    )
    rows = read_review_csv(path)
    assert [r["id"] for r in rows] == ["g001", "g097", "g098"]
    assert rows[0]["notes"] == "Note: 2017–2018 range"


def test_append_adds_a_missing_final_newline_without_touching_content(tmp_path):
    path = tmp_path / "review.csv"
    old = write_csv(path, ",".join(REVIEW_COLUMNS) + "\n" + ",".join(["g001"] + [""] * 11))
    append_review_csv(path, [row("g097")])
    new = path.read_bytes()
    assert new.startswith(old)
    assert [r["id"] for r in read_review_csv(path)] == ["g001", "g097"]


@pytest.mark.parametrize("batch", ["topup1", "topup2"])
def test_every_topup_batch_is_registered(batch):
    assert batch in TOPUPS
