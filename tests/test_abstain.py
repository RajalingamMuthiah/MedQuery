"""Abstention by retrieval confidence (offline) and answer_v2 prompt versioning."""

import math

import numpy as np
import pytest

from medquery.eval import abstain
from medquery.eval.runner import RunConfig, answer_version
from medquery.generation.prompts import ANSWER_HASH, PROMPTS, answer_messages, prompt_id
from tests.test_eval import record


def rows():
    # answerable a, b correct and c incorrect; unanswerable u answered. Signals in SCORES.
    return [
        record("a", correctness="correct"),
        record("b", correctness="correct"),
        record("c", correctness="incorrect"),
        record("u", answerable=False, abstained=False, correctness="incorrect"),
    ]


SCORES = {"a": 0.9, "b": 0.8, "c": 0.3, "u": 0.2}


def test_credit_rewards_correct_answers_and_abstaining_on_unanswerable():
    r = rows()
    assert [abstain.credit(x) for x in r] == [1.0, 1.0, 0.0, 0.0]
    assert abstain.credit(record("v", answerable=False, abstained=True)) == 1.0


def test_threshold_turns_low_confidence_answers_into_abstentions():
    applied = abstain.apply(rows(), SCORES, 0.5)
    assert [x["abstained"] for x in applied] == [False, False, True, True]
    assert applied[2]["judge"] is None and applied[3]["abstained_by_threshold"]
    assert abstain.overall(applied) == pytest.approx(0.75)  # a, b correct; u now abstains
    assert abstain.apply(rows(), SCORES, abstain.NO_THRESHOLD) == rows()


def test_curve_starts_at_full_coverage_and_coverage_never_rises():
    points = abstain.curve(rows(), SCORES)
    assert points[0]["threshold"] == -math.inf and points[0]["coverage"] == 1.0
    covs = [p["coverage"] for p in points]
    assert covs == sorted(covs, reverse=True)


def test_operating_point_maximises_overall_and_ties_go_lower():
    # thresholds 0.3 (drops u) and any value in (0.3, 0.8] both score 0.75 ... first best is 0.3
    t = abstain.operating_point(rows(), SCORES)
    assert t == 0.3
    assert abstain.overall(abstain.apply(rows(), SCORES, t)) == pytest.approx(0.75)


def test_leave_one_out_never_uses_the_held_out_item():
    seen = []
    real = abstain.operating_point

    def spy(records, scores):
        seen.append({r["id"] for r in records})
        return real(records, scores)

    abstain.operating_point = spy
    try:
        loo = abstain.leave_one_out(rows(), SCORES)
    finally:
        abstain.operating_point = real
    assert [s for s in seen] == [{"b", "c", "u"}, {"a", "c", "u"}, {"a", "b", "u"}, {"a", "b", "c"}]
    assert 0.0 <= loo["overall"] <= 1.0


class FakeEmbedder:
    def embed_passages(self, texts):
        return np.stack([[1.0, 0.0] if "match" in t else [0.0, 1.0] for t in texts])

    def embed_query(self, text):
        return np.array([1.0, 0.0])


def test_confidence_is_the_best_cosine_over_the_context_passages():
    recs = [record("a", context=["x", "y"]), record("b", context=["y"])]
    chunks = {"x": {"text": "match"}, "y": {"text": "other"}}
    scores = abstain.confidence(recs, chunks, FakeEmbedder(), lambda c: c["text"])
    assert scores == pytest.approx({"a": 1.0, "b": 0.0})


# --- answer_v2 ---


def test_answer_v2_is_a_separate_version_and_v1_is_unchanged():
    assert prompt_id("answer_v1") == f"answer_v1:{ANSWER_HASH}" == "answer_v1:9cf9329e3ba3"
    assert prompt_id("answer_v2").startswith("answer_v2:") and prompt_id("answer_v2") != prompt_id(
        "answer_v1"
    )
    v2 = PROMPTS["answer_v2"][0]
    for rule in (
        "Add no detail",
        "plain ASCII",
        "units and qualifiers",
        "which part",
        "SUPERSEDED",
    ):
        assert rule in v2
    msgs = answer_messages("q?", [], "answer_v2")
    assert msgs[0]["content"] == v2 and msgs[1]["content"].endswith("Question: q?")


def test_a_changed_prompt_text_needs_a_new_version():
    assert answer_version(RunConfig(answer_prompt=prompt_id("answer_v2"))) == "answer_v2"
    with pytest.raises(ValueError, match="new version"):
        answer_version(RunConfig(answer_prompt="answer_v2:000000000000"))
