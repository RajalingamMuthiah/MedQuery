"""Phase 5 claim verification (offline, fake verifier model)."""

import json

import pytest

from medquery import config
from medquery.eval.compare import verifier_guards
from medquery.eval.runner import CONFIGS, evaluate_item
from medquery.generation.prompts import ABSTAIN_TEXT
from medquery.llm.base import Completion
from medquery.verify.claims import render, split_sentences
from medquery.verify.verifier import VERIFY_HASH, verify, verify_id
from tests.test_eval import claim, record
from tests.test_provider import NullMapper, OneHitRetriever

ANSWER = (
    "Adults should eat less sodium, e.g. under 2,300 mg a day [htn-dash:0004]. "
    "The 2.5 mg dose is daily [tb-ltbi-2020:0020]. "
    "This guidance is from 2003 and has been superseded [htn-jnc7-express:0026]."
)


def test_sentences_keep_their_citations_and_ignore_abbreviations_and_decimals():
    assert split_sentences(ANSWER) == [
        "Adults should eat less sodium, e.g. under 2,300 mg a day [htn-dash:0004].",
        "The 2.5 mg dose is daily [tb-ltbi-2020:0020].",
        "This guidance is from 2003 and has been superseded [htn-jnc7-express:0026].",
    ]
    assert split_sentences("One fact. [a-b:0001] Two facts [a‑b:0002].") == [
        "One fact. [a-b:0001]",
        "Two facts [a‑b:0002].",
    ]


def test_render_drops_sentences_keeps_notes_and_abstains_when_no_fact_survives():
    s = split_sentences(ANSWER)
    fact = [True, True, False]
    assert render(ANSWER, s, [True, True, True], fact) == (ANSWER, False)  # byte-identical
    text, abstained = render(ANSWER, s, [True, False, True], fact)
    assert not abstained and text == f"{s[0]} {s[2]}"
    assert render(ANSWER, s, [False, False, True], fact) == (ABSTAIN_TEXT, True)


class FakeVerifier:
    """Replies with queued JSON (or text) and records each request."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def complete(self, messages, **kw):
        self.requests.append((messages, kw))
        reply = self.replies.pop(0)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return Completion(text=text, model=kw["model"], finish_reason="stop")


def verdict(*sentences):
    return {"sentences": [{"n": n, **s} for n, s in enumerate(sentences, 1)]}


SUPPORTED = {"kind": "fact", "claims": [{"text": "c", "supported": True}]}
UNSUPPORTED = {
    "kind": "fact",
    "claims": [{"text": "a", "supported": True}, {"text": "b", "supported": False}],
}
NOTE = {"kind": "note", "claims": []}


def test_a_sentence_with_any_unsupported_claim_is_removed():
    llm = FakeVerifier(verdict(SUPPORTED, UNSUPPORTED, NOTE))
    v = verify("q?", [], ANSWER, llm, provider="nvidia")
    assert v.removed == ["The 2.5 mg dose is daily [tb-ltbi-2020:0020]."] and not v.abstained
    assert "2.5 mg" not in v.text and "superseded" in v.text
    messages, kw = llm.requests[0]
    assert kw["model"] == config.GENERATOR_MODEL and kw["provider"] == "nvidia"
    assert "1. Adults should eat less sodium" in messages[1]["content"]


def test_invalid_replies_retry_once_then_keep_the_original_answer():
    llm = FakeVerifier("not json", verdict(SUPPORTED, SUPPORTED, NOTE))
    v = verify("q?", [], ANSWER, llm)
    assert not v.failed and v.text == ANSWER and len(llm.requests) == 2
    llm = FakeVerifier("not json", verdict(SUPPORTED))  # wrong sentence count
    v = verify("q?", [], ANSWER, llm)
    assert v.failed and v.text == ANSWER and "expected 3 sentences" in v.error


def test_the_verifier_is_never_the_judge_model():
    with pytest.raises(ValueError, match="judge model"):
        verify("q?", [], ANSWER, FakeVerifier(), model=config.JUDGE_MODEL)
    assert verify_id() == f"verify_v1:{VERIFY_HASH}" and len(VERIFY_HASH) == 12


class RoutingFake:
    """Generator replies with a two-sentence answer; verifier and judge replies are scripted."""

    def __init__(self, verify_reply):
        self.generation, self.verify_reply = [], verify_reply

    def complete(self, messages, **kw):
        system = messages[0]["content"]
        if "check an answer against source passages" in system:
            text = json.dumps(self.verify_reply)
        elif kw["model"] == config.JUDGE_MODEL:
            text = json.dumps(
                {"claims": [], "relevancy": 5, "correctness": "correct", "rationale": "r"}
            )
        else:
            self.generation.append((messages, kw))
            text = "Rifampin is given daily [tb:0001]. It cures everything [tb:0001]."
        return Completion(text=text, model=kw["model"], finish_reason="stop")


ITEM = {
    "id": "g1",
    "split": "dev",
    "topic": "tb",
    "type": "factual",
    "answerable": True,
    "question": "q?",
    "reference_answer": "r",
    "evidence": [],
}


def test_runner_verifies_after_the_same_generation_request_as_the_live_config():
    live = RoutingFake(verdict(SUPPORTED, UNSUPPORTED))
    evaluate_item(
        ITEM, OneHitRetriever(), live, NullMapper(), CONFIGS["hybrid_headers_tables_answer_v2"]
    )
    verified = RoutingFake(verdict(SUPPORTED, UNSUPPORTED))
    rec = evaluate_item(ITEM, OneHitRetriever(), verified, NullMapper(), CONFIGS["live_verify"])
    assert live.generation == verified.generation  # same request, so a cache hit in real runs
    assert rec["answer"] == "Rifampin is given daily [tb:0001]."
    assert rec["verification"]["removed"] == ["It cures everything [tb:0001]."]
    assert rec["verification"]["answer_original"].endswith("It cures everything [tb:0001].")
    assert CONFIGS["live_verify"].verifier == verify_id()
    assert CONFIGS["hybrid_headers_tables_answer_v2"].verifier == ""


def test_guards_count_over_removal_and_missed_claims():
    base = record(
        "g1",
        claims=[
            claim(True) | {"text": "Rifampin is given daily"},
            claim(False) | {"text": "It cures everything"},
        ],
    )
    verified = record("g1", claims=[claim(True), claim(False)]) | {
        "verification": {
            "removed": ["Rifampin is given daily [tb:0001].", "It cures everything [tb:0001]."],
            "failed": False,
            "verify_s": 1.0,
        }
    }
    g = verifier_guards([verified], [base])
    assert (g["removed"], g["removed_supported"], g["removed_unsupported"]) == (2, 1, 1)
    assert g["missed"] == 1 and g["changed"] == 1 and g["verified"] == 1


def test_a_finished_test_run_is_never_run_again(monkeypatch, capsys):
    import medquery.eval.__main__ as cli

    monkeypatch.setattr(cli, "git_state", lambda: {"commit": "x", "dirty": False})
    monkeypatch.setattr(cli, "sha256_file", lambda path: "x")
    monkeypatch.setattr(cli, "chunks_path", lambda corpus: cli.paths.ROOT / "pyproject.toml")
    monkeypatch.setattr(cli, "load_records", lambda path: [{}] * 23)  # a complete test run
    assert cli.main(["run", "--split", "test", "--config", "baseline_nvidia"]) == 1
    assert "already has a complete test run" in capsys.readouterr().out


# --- verify_v2 (live app) ---


def test_verify_v2_is_a_new_version_and_v1_is_unchanged():
    from medquery.verify.verifier import VERIFY_HASH as V1
    from medquery.verify.verifier import VERSIONS

    assert verify_id() == f"verify_v1:{V1}" and verify_id("verify_v1") == verify_id()
    assert verify_id("verify_v2") != verify_id("verify_v1")
    assert VERSIONS["verify_v2"][1]["max_tokens"] == 4000
    assert CONFIGS["live_verify"].verifier == verify_id("verify_v1")  # Phase 5 records


def test_verify_v2_treats_an_empty_reply_as_a_failure_and_reports_unverified():
    from medquery.verify.verifier import statuses

    llm = FakeVerifier("", "")
    v = verify("q?", [], ANSWER, llm, version="verify_v2")
    assert v.failed and "empty reply" in v.error and len(llm.requests) == 2
    assert llm.requests[0][1]["max_tokens"] == 4000
    assert statuses(v, 3) == ["unverified"] * 3
    llm = FakeVerifier("", verdict(SUPPORTED, UNSUPPORTED, NOTE))
    v = verify("q?", [], ANSWER, llm, version="verify_v2")
    assert not v.failed and statuses(v, 3) == ["supported", "unsupported", "note"]
    v1 = verify("q?", [], ANSWER, FakeVerifier("", ""))  # verify_v1: same failure, old message
    assert v1.failed and "empty reply" not in v1.error
