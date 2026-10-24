"""Answer prompts and citation parsing (Phase 4)."""

from medquery.generation.prompts import (
    ANSWER_HASH,
    ANSWER_VERSION,
    extract_citations,
    is_abstention,
)


def test_answer_v1_is_unchanged():
    assert (ANSWER_VERSION, ANSWER_HASH) == ("answer_v1", "9cf9329e3ba3")


def test_citations_with_unicode_hyphens_and_fullwidth_brackets_parse():
    text = (
        "A [dm‑stats‑2020:0014]. B 【dm-stats-2020:0006】. "
        "C [tb‐core–curriculum:0173, htn−dash:0004]."
    )
    assert extract_citations(text) == [
        "dm-stats-2020:0014",
        "dm-stats-2020:0006",
        "tb-core-curriculum:0173",
        "htn-dash:0004",
    ]


def test_malformed_ids_are_still_rejected():
    assert extract_citations("x 【dm-stats-2020:006】 y (dm-stats-2020:0001)") == []


def test_abstention_detection_is_unchanged():
    assert is_abstention("I cannot answer this from the provided documents.")
    assert not is_abstention("The dose is 10 mg [tb-ltbi-2020:0020].")
