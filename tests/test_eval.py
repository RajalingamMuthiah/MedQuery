"""Judge, statistics, metrics, runner, report and judge-review sampling (all offline)."""

import json
import math

import pytest

from medquery.eval import metrics
from medquery.eval.evidence import EvidenceMapper
from medquery.eval.judge import JUDGE_TEMPLATE, JudgeFailed, judge, judge_messages, validate
from medquery.eval.judge_review import agreement_block, probe_units, review_rows, sample
from medquery.eval.report import write_report
from medquery.eval.runner import RunConfig, load_records, run
from medquery.eval.stats import bootstrap_ci, cohen_kappa
from medquery.llm.base import Completion, QuotaExhausted
from medquery.retrieval.dense import Hit

# --- judge prompt and parsing -------------------------------------------------------------

CHUNK = {"id": "tb:0001", "text": "Rifampin 10 mg/kg daily for 4 months; maximum 600 mg."}
HITS = [Hit(CHUNK, 0.9, 1)]
REFERENCE = "Adults take 10 mg/kg daily, up to 600 mg."


def test_reference_answer_comes_after_passages_and_claim_instructions():
    prompt = judge_messages("What is the dose?", HITS, "10 mg/kg [tb:0001].", REFERENCE)[1][
        "content"
    ]
    passages_at = prompt.index(CHUNK["text"])
    claims_at = prompt.index("Instructions for claims")
    support_rule_at = prompt.index('"supported_by_cited"')
    reference_at = prompt.index(REFERENCE)
    assert passages_at < claims_at < support_rule_at < reference_at
    assert "Decide this ONLY from the passages" in prompt
    assert "do NOT use it to decide" in prompt[reference_at - 200 : reference_at]
    assert prompt.index('6. "correctness"') > reference_at  # correctness rule comes after it


GOOD = {
    "claims": [
        {
            "text": "Adults take 10 mg/kg daily.",
            "cited_ids": ["tb:0001"],
            "supported_by_context": True,
            "supported_by_cited": True,
        }
    ],
    "relevancy": 5,
    "correctness": "correct",
    "rationale": "ok",
}


def test_validate_accepts_good_and_rejects_bad_replies():
    assert validate(GOOD).correctness == "correct"
    for broken, problem in [
        ({**GOOD, "relevancy": 7}, "relevancy"),
        ({**GOOD, "correctness": "mostly"}, "correctness"),
        (
            {**GOOD, "claims": [{"text": "x", "supported_by_context": "yes"}]},
            "supported_by_context",
        ),
    ]:
        with pytest.raises(ValueError, match=problem):
            validate(broken)


def test_uncited_claims_get_null_cited_support():
    reply = {
        **GOOD,
        "claims": [
            {
                "text": "x",
                "cited_ids": [],
                "supported_by_context": False,
                "supported_by_cited": True,
            }
        ],
    }
    assert validate(reply).claims[0]["supported_by_cited"] is None


class QueueLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages, **kw):
        self.requests.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return Completion(
            text=text,
            model=kw["model"],
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )


def test_invalid_reply_is_retried_once_as_a_new_request():
    llm = QueueLLM("not json", GOOD)
    verdict, completions = judge("Q?", HITS, "A [tb:0001].", REFERENCE, llm)
    assert verdict.correctness == "correct" and len(completions) == 2
    assert len(llm.requests[1]) == len(llm.requests[0]) + 2  # bad reply + correction appended


def test_two_invalid_replies_raise_with_their_usage_attached():
    with pytest.raises(JudgeFailed) as info:
        judge("Q?", HITS, "A.", REFERENCE, QueueLLM("nope", {"claims": "x"}))
    assert len(info.value.completions) == 2


def test_judge_template_is_versioned_text():
    assert "{reference}" in JUDGE_TEMPLATE and "{passages}" in JUDGE_TEMPLATE


# --- statistics ----------------------------------------------------------------------------


def test_kappa_textbook_values():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # 50 units: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> po 0.7, pe 0.5 -> kappa 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    # independent raters at chance: kappa near 0
    a = ["y", "y", "n", "n"] * 25
    b = ["y", "n", "y", "n"] * 25
    assert cohen_kappa(a, b) == pytest.approx(0.0)
    assert cohen_kappa([], []) is None


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    values = [0.0, 1.0] * 20

    def mean(sample):
        return sum(sample) / len(sample)

    first, second = bootstrap_ci(values, mean), bootstrap_ci(values, mean)
    assert first == second
    point, low, high = first
    assert low <= point <= high and point == 0.5 and 0.3 < low < 0.5 < high < 0.7


# --- metrics on hand-computed records --------------------------------------------------------


def claim(supported, cited=(), cited_ok=None):
    return {
        "text": "c",
        "cited_ids": list(cited),
        "supported_by_context": supported,
        "supported_by_cited": cited_ok if cited else None,
    }


def record(
    rid,
    answerable=True,
    abstained=False,
    retrieved=(),
    context=None,
    gold=(),
    evidence=None,
    claims=(),
    correctness="correct",
    relevancy=5,
    cited=(),
    topic="tb",
):
    retrieved = list(retrieved)
    return {
        "id": rid,
        "split": "dev",
        "topic": topic,
        "type": "factual" if answerable else "unanswerable",
        "answerable": answerable,
        "abstained": abstained,
        "question": f"{rid}?",
        "answer": "a",
        "reference_answer": "r",
        "retrieved_ids": retrieved,
        "context_ids": list(context if context is not None else retrieved[:5]),
        "gold_chunks": list(gold),
        "evidence_map": evidence
        if evidence is not None
        else [{"chunk_ids": [g], "status": "exact"} for g in gold],
        "judge": None
        if abstained
        else {"claims": list(claims), "relevancy": relevancy, "correctness": correctness},
        "judge_failed": False,
        "cited_ids": list(cited),
        "finish_reason": "stop",
        "latency": {"retrieval_s": 0.01, "generation_s": 1.0, "end_to_end_s": 1.01, "wait_s": 0},
        "generation_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "calls": 1,
            "cached": 0,
            "reasoning_tokens": 0,
        },
        "judge_usage": {
            "prompt_tokens": 200,
            "completion_tokens": 50,
            "calls": 1,
            "cached": 0,
            "reasoning_tokens": 0,
        },
        "cost_usd": {"generation": 0.001, "judge": 0.002},
    }


ITEMS = [
    # gold at rank 2 of 10; context = first 5
    record(
        "a",
        retrieved=["x", "g1", "y", "z", "w", "v", "u", "t", "s", "r"],
        gold=["g1"],
        claims=[claim(True, ["g1"], True), claim(False, ["x"], False)],
        cited=["g1", "x"],
        correctness="partial",
        relevancy=3,
    ),
    # two gold chunks, one at rank 1, one at rank 7 (outside the context)
    record(
        "b",
        retrieved=["g2", "a", "b", "c", "d", "e", "g3", "f", "h", "i"],
        gold=["g2", "g3"],
        claims=[claim(True, ["g2"], True), claim(True)],
        cited=["g2", "zz"],
    ),
    record("c", abstained=True, retrieved=["q"] * 10, gold=["g4"]),  # answerable but abstained
    record("d", answerable=False, abstained=True, retrieved=["q"] * 10),  # correct abstention
    record(
        "e",
        answerable=False,
        retrieved=["q"] * 10,
        claims=[claim(False)],
        correctness="incorrect",
        relevancy=1,
    ),  # should have abstained
]


def test_retrieval_metrics_hand_computed():
    assert metrics.recall_at(1)(ITEMS) == pytest.approx((0 + 0.5 + 0) / 3)
    assert metrics.recall_at(5)(ITEMS) == pytest.approx((1 + 0.5 + 0) / 3)
    assert metrics.recall_at(10)(ITEMS) == pytest.approx((1 + 1 + 0) / 3)
    assert metrics.mrr(ITEMS) == pytest.approx((0.5 + 1 + 0) / 3)
    # AP@5: a -> (1/2)/1 = 0.5; b -> (1/1)/min(2,5) = 0.5; c -> 0
    assert metrics.context_precision(ITEMS) == pytest.approx((0.5 + 0.5 + 0) / 3)
    assert metrics.context_recall(ITEMS) == pytest.approx((1 + 0.5 + 0) / 3)


def test_abstention_and_correctness_hand_computed():
    assert metrics.abstention_accuracy(ITEMS) == pytest.approx(3 / 5)  # a, b, d right
    assert metrics.abstention_recall(ITEMS) == pytest.approx(1 / 2)
    assert metrics.false_abstention_rate(ITEMS) == pytest.approx(1 / 3)
    assert metrics.answer_correctness(ITEMS) == pytest.approx((0.5 + 1 + 0) / 3)  # c abstained = 0


def test_claim_metrics_hand_computed():
    # answered + judged: a (2 claims, 1 supported), b (2, 2), e (1, 0)
    assert metrics.faithfulness(ITEMS) == pytest.approx((0.5 + 1 + 0) / 3)
    assert metrics.hallucination_rate(ITEMS) == pytest.approx(2 / 5)
    assert metrics.answer_relevancy(ITEMS) == pytest.approx((0.5 + 1 + 0) / 3)
    assert metrics.citation_accuracy(ITEMS) == pytest.approx(2 / 3)  # cited: a1 ok, a2 no, b1 ok
    assert metrics.uncited_claim_share(ITEMS) == pytest.approx(2 / 5)
    # citations: a -> g1 ok, x ok (both in context); b -> g2 ok, zz not in context
    assert metrics.citation_validity(ITEMS) == pytest.approx(3 / 4)


def test_compute_returns_every_metric_with_an_interval():
    rows = metrics.compute(ITEMS)
    names = {r["metric"] for r in rows}
    assert {"answer_correctness", "mrr", "recall@10", "hallucination_rate"} <= names
    for r in rows:
        if r["value"] is not None and r["ci_low"] is not None:
            assert r["ci_low"] <= r["value"] + 1e-9 and r["value"] <= r["ci_high"] + 1e-9


# --- runner: resume after interruption --------------------------------------------------------


def chunk_record(n: int) -> dict:
    return {
        "id": f"tb:{n:04d}",
        "doc_id": "tb",
        "title": "TB guide",
        "publisher": "CDC",
        "as_of": "2020",
        "superseded": "",
        "section": "Dosing",
        "page_start": 1,
        "page_end": 1,
        "kind": "text",
        "text": "Rifampin 10 mg/kg daily.",
    }


class FakeRetriever:
    def retrieve(self, question, k):
        return [Hit(chunk_record(n), 1 - n / 10, n + 1) for n in range(k)], 0.005


GOLDEN = [
    {
        "id": f"g{n:03d}",
        "split": "dev",
        "topic": "tuberculosis",
        "type": "factual",
        "answerable": True,
        "question": f"Question {n}?",
        "reference_answer": "10 mg/kg daily.",
        "evidence": [{"doc_id": "tb", "page": 1, "quote": "Rifampin 10 mg/kg daily"}],
    }
    for n in range(1, 5)
] + [
    {
        "id": "g999",
        "split": "test",
        "topic": "tuberculosis",
        "type": "factual",
        "answerable": True,
        "question": "Test item?",
        "reference_answer": "x",
        "evidence": [],
    }
]
MAPPER = EvidenceMapper(
    [
        {
            "id": "tb:0000",
            "doc_id": "tb",
            "page_start": 1,
            "page_end": 1,
            "text": "Rifampin 10 mg/kg daily.",
        }
    ]
)


class ScriptedRunLLM:
    """Generator answers with a citation; judge returns GOOD. Optionally fails after N calls."""

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def complete(self, messages, **kw):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise QuotaExhausted("daily quota", 3600)
        text = json.dumps(GOOD) if "evaluator" in messages[0]["content"] else "10 mg/kg [tb:0000]."
        return Completion(
            text=text,
            model=kw["model"],
            finish_reason="stop",
            latency_s=0.5,
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )


def test_run_resumes_after_interruption_and_matches_an_uninterrupted_run(tmp_path):
    config = RunConfig()
    with pytest.raises(QuotaExhausted):  # 2 calls per item: dies during the 2nd item
        run(
            config,
            "dev",
            GOLDEN,
            FakeRetriever(),
            ScriptedRunLLM(fail_after=3),
            MAPPER,
            tmp_path / "r1",
            log=lambda *_: None,
        )
    assert [r["id"] for r in load_records(tmp_path / "r1" / "items.jsonl")] == ["g001"]
    resumed = run(
        config,
        "dev",
        GOLDEN,
        FakeRetriever(),
        ScriptedRunLLM(),
        MAPPER,
        tmp_path / "r1",
        log=lambda *_: None,
    )
    fresh = run(
        config,
        "dev",
        GOLDEN,
        FakeRetriever(),
        ScriptedRunLLM(),
        MAPPER,
        tmp_path / "r2",
        log=lambda *_: None,
    )
    assert [r["id"] for r in resumed] == ["g001", "g002", "g003", "g004"]  # test item skipped
    assert resumed == fresh
    assert json.loads((tmp_path / "r1" / "run.json").read_text())["split"] == "dev"


def test_torn_last_line_is_cut_and_that_item_rerun(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_bytes(b'{"id": "g001"}\n{"id": "g0')
    assert load_records(path) == [{"id": "g001"}]
    assert path.read_bytes() == b'{"id": "g001"}\n'


def test_record_fields_for_metrics(tmp_path):
    [first, *_] = run(
        RunConfig(),
        "dev",
        GOLDEN,
        FakeRetriever(),
        ScriptedRunLLM(),
        MAPPER,
        tmp_path / "r",
        log=lambda *_: None,
    )
    assert len(first["retrieved_ids"]) == 10 and len(first["context_ids"]) == 5
    assert first["gold_chunks"] == ["tb:0000"] and first["evidence_map"][0]["status"] == "exact"
    assert first["cited_ids"] == ["tb:0000"] and first["judge"]["correctness"] == "correct"
    assert first["cost_usd"]["generation"] > 0 and first["latency"]["generation_s"] == 0.5


# --- report --------------------------------------------------------------------------------------

GOLDEN_META = {"version": "v1", "sha256": "abc", "items": 5, "reviewer": "test reviewer"}


def test_report_records_split_commit_dirty_flag_and_unvalidated_judge(tmp_path):
    git = {"commit": "deadbeef", "dirty": True}
    md_path, csv_path, items_path = write_report(
        ITEMS, RunConfig(), "dev", "run123", GOLDEN_META, "chunks-sha", git, None, out_dir=tmp_path
    )
    md = md_path.read_text(encoding="utf-8")
    assert "on the **dev** split" in md
    assert "`deadbeef` (DIRTY working tree)" in md
    assert "NOT YET VALIDATED" in md
    assert "reviewer: test reviewer" in md
    assert "answer_correctness" in csv_path.read_text(encoding="utf-8")
    assert items_path.read_text(encoding="utf-8").count("\n") == len(ITEMS) + 1


def test_agreement_block_reports_kappa_and_marks_stale_labels():
    labels = (
        [{"unit": "claim", "human_label": "supported", "judge_label": "supported"}] * 30
        + [{"unit": "claim", "human_label": "unsupported", "judge_label": "unsupported"}] * 8
        + [{"unit": "claim", "human_label": "unsupported", "judge_label": "supported"}] * 2
        + [{"unit": "answer", "human_label": "correct", "judge_label": "correct"}] * 20
    )
    meta = {"sha256": "x", "judge_prompt": "judge_v1:aaa", "judge_model": "m"}
    block = agreement_block(labels, "judge_v1:aaa", "m", meta)
    assert block["claim"]["n"] == 40 and block["claim"]["agreement"][0] == pytest.approx(0.95)
    assert 0.8 < block["claim"]["kappa"][0] < 0.9 and not block["stale"]
    assert agreement_block(labels, "judge_v2:bbb", "m", meta)["stale"]


def test_report_marks_stale_agreement(tmp_path):
    block = agreement_block(
        [{"unit": "claim", "human_label": "supported", "judge_label": "supported"}],
        "new",
        "m",
        {"sha256": "x", "judge_prompt": "old", "judge_model": "m"},
    )
    md_path, *_ = write_report(
        ITEMS,
        RunConfig(),
        "dev",
        "r",
        GOLDEN_META,
        "c",
        {"commit": "c", "dirty": False},
        block,
        out_dir=tmp_path,
    )
    assert "**STALE:**" in md_path.read_text(encoding="utf-8")


# --- judge-review sampling ---


def many_records():
    records = []
    for n in range(60):
        claims = [claim(n % 4 != 0), claim(True), claim(n % 7 != 0)]
        correctness = ["correct", "correct", "partial", "incorrect", "correct"][n % 5]
        records.append(
            record(
                f"g{n:03d}",
                retrieved=["k"] * 10,
                claims=claims,
                correctness=correctness,
                topic=["tb", "dm", "htn"][n % 3],
            )
        )
    return records


def test_sample_sizes_minority_share_and_hidden_verdicts():
    units = sample(many_records())
    claims = [u for u in units if u["unit"] == "claim"]
    answers = [u for u in units if u["unit"] == "answer"]
    assert len(claims) == 40 and len(answers) == 20
    assert sum(u["judge_label"] == "unsupported" for u in claims) >= math.ceil(40 / 3)
    assert sum(u["judge_label"] != "correct" for u in answers) >= math.ceil(20 / 3)
    assert {u["topic"] for u in units} == {"tb", "dm", "htn"}
    rows, key = review_rows(units, {"k": "passage text"})
    assert set(rows[0]) == {
        "row_id",
        "unit",
        "topic",
        "question",
        "passages",
        "answer",
        "reference_answer",
        "claim",
        "human_label",
        "notes",
    }  # no verdict
    assert all(r["human_label"] == "" for r in rows)
    assert set(key) == {r["row_id"] for r in rows}
    assert {k["judge_label"] for k in key.values()} >= {"supported", "unsupported", "correct"}


def test_sample_is_deterministic():
    first = [(u["unit"], u["record"]["id"], u["claim_index"]) for u in sample(many_records())]
    second = [(u["unit"], u["record"]["id"], u["claim_index"]) for u in sample(many_records())]
    assert first == second


def test_test_split_run_is_refused_on_a_dirty_tree(monkeypatch, capsys):
    import medquery.eval.__main__ as cli

    monkeypatch.setattr(cli, "git_state", lambda: {"commit": "x", "dirty": True})
    assert cli.main(["run", "--split", "test"]) == 1
    assert "clean, committed git tree" in capsys.readouterr().out


# --- mismatched-context probes ---


class ProbeJudge:
    """Judge replies: every claim unsupported (the passages shown belong to another question)."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        reply = {
            "claims": [
                {
                    "text": "c1",
                    "cited_ids": [],
                    "supported_by_context": False,
                    "supported_by_cited": None,
                },
                {
                    "text": "c2",
                    "cited_ids": [],
                    "supported_by_context": False,
                    "supported_by_cited": None,
                },
            ],
            "relevancy": 3,
            "correctness": "partial",
            "rationale": "r",
        }
        return Completion(text=json.dumps(reply), model=kw["model"], finish_reason="stop")


def test_probes_use_other_topic_passages_and_stop_when_enough():
    records = many_records()
    chunks = {
        "k": {
            "id": "k",
            "text": "passage",
            "title": "t",
            "publisher": "p",
            "as_of": "2020",
            "superseded": "",
            "page_start": 1,
            "page_end": 1,
            "section": "",
        }
    }
    judge_llm = ProbeJudge()
    units, calls = probe_units(records, judge_llm, chunks, need_unsupported=5)
    assert judge_llm.calls == 3  # 2 unsupported claims per probe: stops once 5 are reached
    assert len(calls) == 3 and all(u["probe"] and u["judge_label"] == "unsupported" for u in units)
    by_id = {r["id"]: r for r in records}
    assert all(by_id[u["donor_id"]]["topic"] != u["topic"] for u in units)


def test_probes_fill_only_the_unsupported_share():
    records = many_records()
    probes = [
        {
            "unit": "claim",
            "record": records[0],
            "claim_index": n,
            "claim": f"p{n}",
            "topic": "tb",
            "judge_label": "unsupported",
            "context_ids": ["k"],
            "probe": True,
            "donor_id": records[1]["id"],
        }
        for n in range(30)
    ]
    natural_supported_only = [
        {**r, "judge": {**r["judge"], "claims": [claim(True)] * 3}} for r in records
    ]
    units = sample(natural_supported_only, probes=probes)
    claims = [u for u in units if u["unit"] == "claim"]
    assert len(claims) == 40
    assert sum(u["probe"] for u in claims) == math.ceil(40 / 3)  # minority third from probes
    assert all(not u["probe"] for u in claims if u["judge_label"] == "supported")
    rows, key = review_rows(units, {"k": "passage text"})
    assert sum(k["probe"] for k in key.values()) == math.ceil(40 / 3)
