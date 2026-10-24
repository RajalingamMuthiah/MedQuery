"""Phase 4 failure diagnosis and table-chunk statistics (offline)."""

import pytest

from medquery.eval import diagnose
from medquery.ingest.tables import caption, is_continuation, table_stats
from tests.test_eval import claim, record


def test_findings_cover_wrong_answers_unsupported_claims_and_false_abstentions():
    rows = [
        record("g1", claims=[claim(True), claim(False)]),
        record("g2", correctness="partial", claims=[claim(True)]),
        record("g3", abstained=True),
        record("g4", answerable=False, abstained=True),
        record("g5", claims=[claim(True)]),
    ]
    found = diagnose.findings(rows)
    assert [(f["item_id"], f["finding"], f["claim_index"]) for f in found] == [
        ("g1", "unsupported_claim", "1"),
        ("g2", "partial", ""),
        ("g3", "false_abstention", ""),
    ]


def test_every_finding_needs_exactly_one_known_cause(tmp_path):
    found = diagnose.findings([record("g1", claims=[claim(False)])])
    path = tmp_path / "causes.csv"
    path.write_text(
        "run_id,item_id,finding,claim_index,cause,note\nr,g1,unsupported_claim,0,wrong_citation,n\n",
        encoding="utf-8",
    )
    joined = diagnose.join(found, diagnose.load_causes("r", path))
    assert joined[0]["cause"] == "wrong_citation"
    with pytest.raises(diagnose.DiagnosisError, match="without a cause"):
        diagnose.join(
            found + diagnose.findings([record("g2", abstained=True)]),
            diagnose.load_causes("r", path),
        )
    path.write_text(
        "run_id,item_id,finding,claim_index,cause,note\nr,g1,unsupported_claim,0,made_up,n\n",
        encoding="utf-8",
    )
    with pytest.raises(diagnose.DiagnosisError, match="unknown cause"):
        diagnose.load_causes("r", path)


def test_committed_causes_use_known_labels_only():
    for run in ("a14cd47892bd",):
        causes = diagnose.load_causes(run)
        assert causes and {c["cause"] for c in causes.values()} <= set(diagnose.CAUSES)


def table(cid, page, text, doc="d"):
    return {
        "id": cid,
        "doc_id": doc,
        "kind": "table",
        "page_start": page,
        "page_end": page,
        "text": text,
    }


T1 = (
    "Table 1. Counts by age, in millions\n\n"
    "| Age | Number |\n|---|---|\n| | in millions |\n| 18-44 | 3 .6 |"
)
T1_CONT = "| 45-64 | 11.7 |\n|---|---|\n| 65+ | 11.5 |"
T2 = "Table 2. Something that is cut\n\n| a | | b |\n|---|---|---|\n| 1 | | 2 |"


def test_table_stats_count_each_problem():
    chunks = [
        table("d:1", 4, T1),
        table("d:2", 5, T1_CONT),
        table("d:3", 7, T2),
        {
            "id": "d:4",
            "doc_id": "d",
            "kind": "text",
            "page_start": 7,
            "page_end": 7,
            "text": "prose",
        },
    ]
    s = table_stats(chunks)
    assert s["table_chunks"] == 3 and s["chunks"] == 4
    assert s["no_caption"] == 1  # the continuation
    assert s["caption_cut_mid_sentence"] == 2  # neither caption ends with . : or )
    assert s["continuation_without_caption_or_header"] == 1
    assert s["multi_row_header"] == 1 and s["all_empty_columns"] == 1 and s["split_decimals"] == 1
    assert caption(T1) == "Table 1. Counts by age, in millions" and caption(T1_CONT) == ""


def test_continuation_needs_next_page_same_doc_and_same_width():
    a = table("d:1", 4, T1)
    assert is_continuation(a, table("d:2", 5, T1_CONT))
    assert not is_continuation(a, table("d:2", 6, T1_CONT))  # not the next page
    assert not is_continuation(a, table("e:2", 5, T1_CONT, doc="e"))  # another document
    assert not is_continuation(a, table("d:2", 5, "| x | y | z |"))  # different width
    assert not is_continuation(a, table("d:2", 5, T2))  # has its own caption
