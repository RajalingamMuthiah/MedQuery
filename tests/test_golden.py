import json
import re
from collections import Counter

import numpy as np
import pytest

from medquery.eval.__main__ import append_review, next_id_number, review_has_edits
from medquery.eval.evidence import Evidence, EvidenceMapper
from medquery.eval.golden import (
    Item,
    check_evidence,
    check_question,
    copied_spans,
    format_evidence,
    is_duplicate,
    parse_evidence,
    read_review_csv,
    review_row,
    shared_ngram,
    write_review_csv,
)
from medquery.eval.golden_gen import (
    GEN_HASH,
    SUPERSEDED_V2_HASH,
    KeywordScan,
    SourcePage,
    TopupGroup,
    eligible,
    generate,
    parse_json,
    plan_tasks,
    to_items,
    topup,
)
from medquery.ingest.manifest import Document, Manifest
from medquery.llm.base import Completion
from medquery.retrieval.dense import Retriever

PAGE = (
    "Four months of daily rifampin is a preferred regimen for latent tuberculosis infection. "
    "Adults receive 10 mg/kg per day with a maximum dose of 600 mg. "
) * 4


def test_shared_ngram_detects_copied_phrases():
    copied = "Is four months of daily rifampin is a preferred option?"
    assert shared_ngram(copied, PAGE) == "four months of daily rifampin is"
    own_words = "How long does the preferred rifampin course for latent TB last?"
    assert shared_ngram(own_words, PAGE) is None


@pytest.mark.parametrize(
    ("question", "ok"),
    [
        ("What is the maximum daily rifampin dose for adults with LTBI?", True),
        ("What is the maximum daily rifampin dose for adults", False),  # no question mark
        ("Dose?", False),  # too short
    ],
)
def test_check_question(question, ok):
    assert (check_question(question) is None) is ok


def test_duplicate_detection():
    accepted = ["What is the maximum daily rifampin dose for adults with LTBI?"]
    assert is_duplicate(
        "What is the maximum daily dose of rifampin for adults with LTBI?", accepted
    )
    assert not is_duplicate("Which drug is given once weekly with isoniazid?", accepted)


CHUNK = {"id": "tb:0001", "doc_id": "tb", "page_start": 8, "page_end": 8, "text": PAGE}
MAPPER = EvidenceMapper([CHUNK])


def test_check_evidence_accepts_verbatim_quotes():
    ev = [Evidence("tb", 8, "Adults receive 10 mg/kg per day with a maximum dose of 600 mg.")]
    assert check_evidence(ev, {("tb", 8): PAGE}, MAPPER) is None


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        ([Evidence("tb", 8, "Adults receive 15 mg/kg per day with a maximum")], "not found"),
        ([Evidence("tb", 9, "Adults receive 10 mg/kg per day")], "not a source page"),
        ([Evidence("tb", 8, "rifampin")], "quote has 1 words"),
        ([], "0 evidence quotes"),
    ],
)
def test_check_evidence_rejections(evidence, reason):
    assert reason in check_evidence(evidence, {("tb", 8): PAGE}, MAPPER)


def test_quote_on_page_but_in_no_chunk_is_rejected():
    mapper = EvidenceMapper([{**CHUNK, "text": "unrelated text"}])
    ev = [Evidence("tb", 8, "Adults receive 10 mg/kg per day with a maximum dose of 600 mg.")]
    assert "in no chunk" in check_evidence(ev, {("tb", 8): PAGE}, mapper)


def test_evidence_cell_round_trip():
    evidence = [
        Evidence("tb", 8, 'Adults receive 10 mg/kg ("max" 600 mg)'),
        Evidence("tb", 9, "x y z w"),
    ]
    assert parse_evidence(format_evidence(evidence)) == evidence
    with pytest.raises(ValueError, match="doc_id \\| page"):
        parse_evidence("tb page 8 quote")


def test_review_csv_round_trip_with_bom_and_symbols(tmp_path):
    item = Item(
        "g001",
        "What is the dose for ≥50 kg?",
        "900 mg [µ]",
        [Evidence("tb", 8, "≥50.0 kg, 900 mg")],
        "table",
        "tuberculosis",
        True,
    )
    path = tmp_path / "review.csv"
    write_review_csv(path, [review_row(item, "Doc, p.8")])
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # BOM for Excel
    [row] = read_review_csv(path)
    assert row["question"] == item.question
    assert row["decision"] == "keep" and row["reviewed"] == ""
    assert parse_evidence(row["evidence"]) == item.evidence
    assert not review_has_edits(path)
    row["reviewed"] = "y"
    write_review_csv(path, [row])
    assert review_has_edits(path)


def page(doc_id, number, text=PAGE, source="text", conf=None, tables=0):
    return SourcePage(doc_id, number, text, source, conf, tables)


def test_eligibility_rules():
    assert eligible(page("tb", 8), {"text"})
    assert not eligible(page("tb", 2), {"text"})  # cover and front matter
    assert not eligible(page("tb", 8), {"text", "quiz"})
    assert not eligible(page("tb", 8), set())  # no chunk covers the page
    assert not eligible(page("tb", 8, "short page"), {"text"})
    refs = "Smith J et al. 2019. " * 5 + PAGE
    assert not eligible(page("tb", 8, refs), {"text"})
    numbers = " ".join(["1,234 56.7 89 12"] * 40)
    assert not eligible(page("tb", 8, numbers, source="ocr", conf=90), {"text"})
    assert eligible(page("tb", 8, PAGE, source="ocr", conf=90), {"text"})
    assert not eligible(page("tb", 8, PAGE, source="ocr", conf=60), {"text"})


def doc(doc_id, topic, superseded=""):
    return Document(
        doc_id,
        f"Title {doc_id}",
        "CDC",
        2020,
        topic,
        f"https://x/{doc_id}.pdf",
        "Public domain",
        "e",
        "2020",
        superseded,
    )


DOCS = {
    "tb": doc("tb", "tuberculosis"),
    "dm": doc("dm", "diabetes"),
    "htn": doc("htn", "hypertension"),
    "jnc": doc("jnc", "hypertension", superseded="Superseded."),
}


def corpus():
    pages, chunks = [], []
    for doc_id in DOCS:
        for n in range(3, 30):
            pages.append(page(doc_id, n, tables=1 if n % 5 == 0 else 0))
            kind = "table" if n % 5 == 0 else "text"
            chunks.append(
                {
                    "id": f"{doc_id}:{n:04d}",
                    "doc_id": doc_id,
                    "page_start": n,
                    "page_end": n,
                    "kind": kind,
                    "text": PAGE,
                }
            )
    return pages, chunks


def test_plan_is_deterministic_and_respects_types():
    pages, chunks = corpus()
    first = plan_tasks(pages, chunks, DOCS)
    second = plan_tasks(pages, chunks, DOCS)
    assert [(t.type, [(p.doc_id, p.page) for p in t.pages]) for t in first] == [
        (t.type, [(p.doc_id, p.page) for p in t.pages]) for t in second
    ]
    for task in first:
        superseded_doc = any(DOCS[p.doc_id].superseded for p in task.pages)
        assert superseded_doc == (task.type == "superseded")
        if task.type == "multi-chunk":
            a, b = task.pages
            assert a.doc_id == b.doc_id and b.page == a.page + 1
        if task.type == "table":
            assert task.pages[0].tables
    pages_used = [(p.doc_id, p.page) for t in first for p in t.pages]
    assert len(pages_used) == len(set(pages_used))  # no page is reused


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('{"items": []}') == {"items": []}
    assert parse_json('Here you go:\n```json\n{"items": [1]}\n```') == {"items": [1]}


def test_keyword_scan_ranks_by_term_overlap():
    chunks = [
        {"id": "a", "text": "rifampin dosing for children"},
        {"id": "b", "text": "insulin pumps"},
        {"id": "c", "text": "rifampin hepatotoxicity monitoring children adults"},
    ]
    scan = KeywordScan(chunks)
    assert [c["id"] for c in scan.top("rifampin monitoring in children?", 2, set())] == ["c", "a"]
    assert [c["id"] for c in scan.top("rifampin", 5, {"a"})] == ["c"]


class FakeEmbedder:
    model_name = "fake"

    def embed_passages(self, texts):
        return np.ones((len(texts), 3), dtype=np.float32)

    def embed_query(self, text):
        return np.ones(3, dtype=np.float32)


UNANSWERABLE_Q = "What rifampin dose is advised for pregnant teenagers on dialysis?"
GOOD_Q = "How long does the preferred rifampin course for latent TB last?"
COPIED_Q = "Is four months of daily rifampin is a preferred option?"
QUOTE = "Four months of daily rifampin is a preferred regimen"


CLEAN_REWRITE = "Which rifampin course length is preferred when treating LTBI?"


class ScriptedLLM:
    """Generation replies quote the page shown in the prompt; verification is a fixed verdict;
    rewrites return `rewrite`."""

    def __init__(self, verdict, rewrite=CLEAN_REWRITE):
        self.verdict = verdict
        self.rewrite = rewrite
        self.rewrite_prompts: list[str] = []

    def complete(self, messages, **kw):
        user = messages[-1]["content"]
        if "Rewrite this question" in user:
            self.rewrite_prompts.append(user)
            reply = {"question": self.rewrite}
        elif "Does any passage contain" in user:
            reply = self.verdict
        elif "could plausibly ask next" in user:
            reply = {"items": [{"question": UNANSWERABLE_Q, "why_not_answerable": "subgroup"}]}
        else:
            shown = int(re.search(r"Page (\d+):", user)[1])
            evidence = [{"page": shown, "quote": QUOTE}]
            reply = {
                "items": [
                    {"question": GOOD_Q, "reference_answer": "Four months.", "evidence": evidence},
                    {"question": COPIED_Q, "reference_answer": "Yes.", "evidence": evidence},
                ]
            }
        return Completion(text=json.dumps(reply), model=kw["model"], finish_reason="stop")


def test_generate_validates_and_verifies_end_to_end(tmp_path):
    pages = [page("tb", n) for n in range(3, 20)]
    chunks = [
        {**CHUNK, "id": f"tb:{n:04d}", "page_start": n, "page_end": n, "kind": "text"}
        for n in range(3, 20)
    ]
    manifest = Manifest([DOCS["tb"]], [])
    retriever = Retriever(FakeEmbedder(), chunks, embed_cache=tmp_path / "e.npz")

    found = generate(
        manifest,
        pages,
        chunks,
        retriever,
        ScriptedLLM({"answer_found": True, "partial": False}),
        log=lambda *_: None,
    )
    reasons = [c["reject_reason"] for c in found.candidates]
    assert "answer found in corpus by verification check" in reasons
    rewritten = [c for c in found.candidates if c.get("original_question") == COPIED_Q]
    assert rewritten and rewritten[0]["question"] == CLEAN_REWRITE
    assert rewritten[0]["status"] == "accepted"
    assert rewritten[0]["copy_reason"].startswith("question copies 6 words")
    accepted = [c for c in found.candidates if c["status"] == "accepted"]
    assert accepted and all(c["answerable"] for c in accepted)
    assert all(c["gen_model"] == "qwen/qwen3.8-27b" for c in found.candidates)

    clean = generate(
        manifest,
        pages,
        chunks,
        retriever,
        ScriptedLLM({"answer_found": False, "partial": False}),
        log=lambda *_: None,
    )
    unanswerable = [
        c for c in clean.candidates if not c["answerable"] and c["status"] == "accepted"
    ]
    assert unanswerable and unanswerable[0]["evidence"] == []
    assert unanswerable[0]["verification"]["checked_ids"]

    items = to_items(clean.candidates)
    assert [i.id for i in items] == [f"g{n:03d}" for n in range(1, len(items) + 1)]


def test_a_rewrite_that_still_copies_is_rejected_after_one_try(tmp_path):
    pages = [page("tb", n) for n in range(3, 20)]
    chunks = [
        {**CHUNK, "id": f"tb:{n:04d}", "page_start": n, "page_end": n, "kind": "text"}
        for n in range(3, 20)
    ]
    retriever = Retriever(FakeEmbedder(), chunks, embed_cache=tmp_path / "e.npz")
    llm = ScriptedLLM({"answer_found": False, "partial": False}, rewrite=COPIED_Q)
    result = generate(
        Manifest([DOCS["tb"]], []), pages, chunks, retriever, llm, log=lambda *_: None
    )
    copied = [c for c in result.candidates if c.get("original_question") == COPIED_Q]
    assert copied
    assert all(c["reject_reason"].startswith("after one rewrite: question copies") for c in copied)
    assert '"four months of daily rifampin is a preferred"' in llm.rewrite_prompts[0]


def test_copied_spans_merges_overlapping_ngrams():
    question = "Is four months of daily rifampin is a preferred option for adults?"
    assert copied_spans(question, PAGE) == ["four months of daily rifampin is a preferred"]
    assert copied_spans("How long is the rifampin course?", PAGE) == []


# --- top-up ---


class RecordingLLM(ScriptedLLM):
    def __init__(self, verdict):
        super().__init__(verdict)
        self.author_prompts: list[str] = []

    def complete(self, messages, **kw):
        if (
            kw["model"] == "qwen/qwen3.8-27b"
            and "Rewrite this question" not in messages[-1]["content"]
        ):
            self.author_prompts.append(messages[-1]["content"])
        return super().complete(messages, **kw)


def topup_corpus():
    docs = {"tb": DOCS["tb"], "jnc": DOCS["jnc"]}
    pages, chunks = [], []
    for doc_id in docs:
        for n in range(3, 15):
            text = PAGE + (
                " Stage 1 is 140 mmHg; drug therapy and lifestyle goals." if doc_id == "jnc" else ""
            )
            pages.append(page(doc_id, n, text))
            chunks.append(
                {
                    "id": f"{doc_id}:{n:04d}",
                    "doc_id": doc_id,
                    "page_start": n,
                    "page_end": n,
                    "kind": "text",
                    "text": text,
                }
            )
    return Manifest(list(docs.values()), []), pages, chunks


def test_topup_skips_used_pages_meets_targets_and_versions_superseded_prompt(tmp_path):
    manifest, pages, chunks = topup_corpus()
    retriever = Retriever(FakeEmbedder(), chunks, embed_cache=tmp_path / "e.npz")
    existing = [{"source_pages": [{"doc_id": "tb", "page": n}]} for n in range(3, 9)]
    groups = (
        TopupGroup("tb-unanswerable", "unanswerable", "tuberculosis", ("tb",), 1, 4, 2),
        TopupGroup("jnc7-substantive", "superseded", "hypertension", ("jnc",), 2, 6, 1),
    )
    llm = RecordingLLM({"answer_found": False, "partial": False})
    result = topup(
        manifest, pages, chunks, retriever, llm, existing, [], groups=groups, log=lambda *_: None
    )

    used_before = {("tb", n) for n in range(3, 9)}
    for c in result.candidates:
        assert (c["source_pages"][0]["doc_id"], c["source_pages"][0]["page"]) not in used_before
        assert c["batch"] == "topup1"
    by_group = Counter(c["topup_group"] for c in result.candidates if c["status"] == "accepted")
    assert by_group["tb-unanswerable"] == 1  # stopped at its target
    superseded = [c for c in result.candidates if c["type"] == "superseded"]
    assert superseded and all(
        c["gen_prompt"].startswith("golden_gen_v1+superseded_v2:") for c in superseded
    )
    assert any("SUBSTANTIVE clinical guidance" in p for p in llm.author_prompts)
    assert SUPERSEDED_V2_HASH != GEN_HASH


def test_topup_checks_duplicates_against_existing_questions(tmp_path):
    manifest, pages, chunks = topup_corpus()
    retriever = Retriever(FakeEmbedder(), chunks, embed_cache=tmp_path / "e.npz")
    groups = (TopupGroup("tb-unanswerable", "unanswerable", "tuberculosis", ("tb",), 1, 2, 1),)
    llm = ScriptedLLM({"answer_found": False, "partial": False})
    result = topup(
        manifest,
        pages,
        chunks,
        retriever,
        llm,
        [],
        [UNANSWERABLE_Q],
        groups=groups,
        log=lambda *_: None,
    )
    assert {c["reject_reason"] for c in result.candidates} == {
        "near-duplicate of an accepted question"
    }


def test_append_review_keeps_existing_rows_and_continues_ids():
    existing = [{"id": "g001", "question": "a"}, {"id": "g086", "question": "b"}]
    assert next_id_number(existing) == 87
    new = [{"id": "g087", "question": "c"}]
    assert append_review(existing, new) == existing + new
    with pytest.raises(ValueError, match="collide"):
        append_review(existing, [{"id": "g086", "question": "x"}])


def test_to_items_can_start_at_an_offset_without_sorting():
    candidates = [
        {
            "status": "accepted",
            "topic": "tuberculosis",
            "type": "unanswerable",
            "question": "q1",
            "reference_answer": "r",
            "evidence": [],
            "answerable": False,
            "source_pages": [{"doc_id": "tb", "page": 3}],
            "gen_model": "m",
            "generator_note": "",
        },
        {
            "status": "accepted",
            "topic": "hypertension",
            "type": "superseded",
            "question": "q2",
            "reference_answer": "r",
            "evidence": [],
            "answerable": True,
            "source_pages": [{"doc_id": "jnc", "page": 4}],
            "gen_model": "m",
            "generator_note": "",
        },
    ]
    items = to_items(candidates, start=87, sort=False)
    assert [(i.id, i.question) for i in items] == [("g087", "q1"), ("g088", "q2")]
