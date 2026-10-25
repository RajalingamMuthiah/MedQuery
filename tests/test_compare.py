"""Ablation comparison: paired bootstrap, multi-chunk breakout, truncation, diagnosis, and the
fresh judge-review sample that excludes already-labelled units."""

from dataclasses import replace

import pytest

from medquery.eval import compare, metrics
from medquery.eval.compare import (
    ablation_markdown,
    diagnose,
    multi_chunk,
    paired_bootstrap,
    truncation,
    write_ablation,
)
from medquery.eval.judge_review import (
    _key,
    labelled_units,
    probe_units,
    review_paths,
    review_rows,
    sample,
    write_review,
)
from medquery.eval.runner import CONFIGS
from tests.test_eval import claim, many_records, record

GIT = {"commit": "abc", "dirty": False}
GOLDEN = {"version": "v1", "sha256": "g" * 64}


def records(n=20, correct=True, abstain_every=0):
    out = []
    for i in range(n):
        abstained = bool(abstain_every) and i % abstain_every == 0
        out.append(
            record(
                f"g{i:03d}",
                abstained=abstained,
                retrieved=[f"c{i}"] + ["x"] * 9 if correct else ["x"] * 10,
                gold=[f"c{i}"],
                correctness="correct" if correct else "incorrect",
            )
        )
    return out


# --- paired bootstrap ---


def test_identical_configs_differ_by_exactly_zero():
    rows = records()
    assert paired_bootstrap(rows, rows, metrics.answer_correctness) == (0.0, 0.0, 0.0)


def test_a_constant_improvement_has_the_right_sign_and_excludes_zero():
    worse, better = records(correct=False), records(correct=True)
    d, low, high = paired_bootstrap(worse, better, metrics.answer_correctness)
    assert d == pytest.approx(1.0) and low == pytest.approx(1.0) and high == pytest.approx(1.0)
    d, low, high = paired_bootstrap(better, worse, metrics.recall_at(5))
    assert d == pytest.approx(-1.0) and high < 0


def test_paired_bootstrap_is_deterministic_and_pairs_by_question():
    a = records(abstain_every=3)
    b = records(abstain_every=4)
    first = paired_bootstrap(a, b, metrics.false_abstention_rate)
    assert first == paired_bootstrap(a, b, metrics.false_abstention_rate)
    assert first == paired_bootstrap(list(reversed(a)), b, metrics.false_abstention_rate)
    assert first[1] <= first[0] <= first[2]


def test_configs_on_different_questions_cannot_be_paired():
    with pytest.raises(ValueError, match="different questions"):
        paired_bootstrap(records(5), records(6), metrics.mrr)


# --- breakouts ---


def multi(rid, context, gold=("a", "b"), correctness="partial", abstained=False):
    r = record(
        rid,
        retrieved=list(context) + ["x"] * (10 - len(context)),
        context=list(context),
        gold=gold,
        correctness=correctness,
        abstained=abstained,
    )
    r["type"] = "multi-chunk"
    return r


def test_multi_chunk_breakout_counts_both_gold_in_context():
    rows = [
        multi("m1", ["a", "x", "x", "x", "x"]),  # second gold chunk missing
        multi("m2", ["a", "x", "x", "x", "x", "x", "x", "b"], correctness="correct"),
        record("s1", retrieved=["c"] * 10, gold=["c"]),  # single-chunk: left out
    ]
    m = multi_chunk(rows)
    assert m["n"] == 2 and m["both_gold_in_context"] == 1
    assert m["answer_correctness"][0] == pytest.approx(0.75)
    assert m["context_recall"][0] == pytest.approx(0.75)
    assert m["recall@5"][0] == pytest.approx(0.5) and m["recall@10"][0] == pytest.approx(0.75)


def test_diagnosis_places_the_best_gold_rank_relative_to_k():
    rows = [
        record("f1", abstained=True, retrieved=["x"] * 6 + ["g"] + ["x"] * 3, gold=["g"]),
        record("f2", abstained=True, retrieved=["x"] * 10, gold=["g"]),
        record("w1", retrieved=["g"] + ["x"] * 9, gold=["g"], correctness="partial"),
    ]
    d = diagnose(rows, k=5)
    assert d["false abstention"] == {"beyond k": 1, "not in top 10": 1}
    assert d["incorrect/partial"] == {"in context": 1}
    assert diagnose(rows, k=8)["false abstention"] == {"in context": 1, "not in top 10": 1}


def with_stats(rows, pairs, cut):
    return [r | {"retrieval_stats": {"rerank_pairs": pairs, "rerank_truncated": cut}} for r in rows]


def test_truncation_share_and_the_report_note_above_ten_percent():
    assert truncation(records(3)) is None
    low = with_stats(records(4), 20, 1)  # 5%
    high = with_stats(records(4), 20, 4)  # 20%
    assert truncation(high) == {"pairs": 80, "truncated": 16, "share": 0.2}
    runs = [
        (CONFIGS["hybrid_rerank"], "r1", low),
        (CONFIGS["hybrid_rerank_headers"], "r2", high),
    ]
    md, _ = ablation_markdown(runs, "dev", GIT, GOLDEN, "c" * 64, "2026-10-22 00:00")
    notes = [line for line in md.splitlines() if line.startswith("**Note:**")]
    assert len(notes) == 1 and "hybrid_rerank_headers" in notes[0] and "20%" in notes[0]
    assert "| `hybrid_rerank` | 80 | 4 | 5.0% |" in md


def test_ablation_report_rows_and_real_verdicts(tmp_path):
    base, better = records(correct=False), records(correct=True)
    runs = [(CONFIGS["baseline"], "b1", base), (CONFIGS["hybrid"], "h1", better)]
    md, rows = ablation_markdown(runs, "dev", GIT, GOLDEN, "c" * 64, "2026-10-22 00:00")
    assert "`abc` (clean working tree)" in md
    assert "| `hybrid` vs `baseline` | answer_correctness | +1.000 |" in md
    assert "**yes**" in md and "BAAI" not in md.split("## Metrics")[0].split("baseline")[1][:40]
    assert {r["metric"] for r in rows} >= {"answer_correctness", "diff_vs:baseline:mrr"}
    assert "## Multi-chunk questions" in md and "## Latency and cost" in md
    written = write_ablation(runs, "dev", GIT, GOLDEN, "c" * 64, out_dir=tmp_path)
    assert [p.suffix for p in written] == [".md", ".csv"] and "_ablation_dev_" in written[0].name


def test_ablation_refuses_the_test_split(tmp_path):
    runs = [(CONFIGS["baseline"], "b1", records())]
    with pytest.raises(ValueError, match="dev"):
        write_ablation(runs, "test", GIT, GOLDEN, "c" * 64, out_dir=tmp_path)


def test_ablation_cli_refuses_test_and_dirty_trees(monkeypatch, capsys):
    import medquery.eval.__main__ as cli

    assert cli.main(["ablation", "--split", "test", "--configs", "baseline"]) == 1
    assert "needs --final" in capsys.readouterr().out
    monkeypatch.setattr(cli, "git_state", lambda: {"commit": "x", "dirty": True})
    assert cli.main(["ablation", "--split", "dev", "--configs", "baseline"]) == 1
    assert "clean, committed" in capsys.readouterr().out


def test_reranker_licence_is_in_the_configs_table():
    runs = [(CONFIGS["hybrid"], "h", records()), (CONFIGS["hybrid_rerank"], "r", records())]
    md, _ = ablation_markdown(runs, "dev", GIT, GOLDEN, "c" * 64, "t")
    assert "`BAAI/bge-reranker-base` (MIT)" in md
    assert compare.TRUNCATION_NOTE == 0.10


# --- fresh judge-review sample ---


def distinct_records():
    """many_records() with distinct claim and answer texts, so exclusion by text is testable."""
    out = []
    for r in many_records():
        claims = [c | {"text": f"{r['id']} claim {n}"} for n, c in enumerate(r["judge"]["claims"])]
        out.append(r | {"answer": f"answer of {r['id']}", "judge": r["judge"] | {"claims": claims}})
    return out


def test_fresh_sample_excludes_units_labelled_in_an_earlier_review(tmp_path):
    rows_all = distinct_records()
    first = sample(rows_all, 40, 20)
    rows, key = review_rows(first, {"k": "passage"})
    write_review(rows, key, {}, review_paths("v1", tmp_path))
    claims, answers = labelled_units("v1", tmp_path)
    assert len(claims) == 40 and len(answers) == 20
    fresh = sample(rows_all, 25, 10, exclude_claims=claims, exclude_answers=answers)
    assert sum(u["unit"] == "claim" for u in fresh) == 25
    assert sum(u["unit"] == "answer" for u in fresh) == 10
    assert not {_key(u["claim"]) for u in fresh if u["unit"] == "claim"} & claims
    assert (
        not {
            (u["record"]["id"], _key(u["record"]["answer"])) for u in fresh if u["unit"] == "answer"
        }
        & answers
    )
    assert _key("  The  Claim\n") == "the claim"


def test_an_answer_that_changed_is_not_excluded():
    rows_all = distinct_records()
    excluded = {(r["id"], _key("an older answer")) for r in rows_all}
    fresh = sample(rows_all, 0, 10, exclude_answers=excluded)
    assert sum(u["unit"] == "answer" for u in fresh) == 10


class CountingJudge:
    def __init__(self):
        self.models = []

    def complete(self, messages, **kw):
        import json

        from medquery.llm.base import Completion

        self.models.append(kw["model"])
        reply = {
            "claims": [
                {
                    "text": t,
                    "cited_ids": [],
                    "supported_by_context": False,
                    "supported_by_cited": None,
                }
                for t in ("old claim", "new claim")
            ],
            "relevancy": 3,
            "correctness": "partial",
            "rationale": "r",
        }
        return Completion(text=json.dumps(reply), model=kw["model"], finish_reason="stop")


CHUNK = {
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


def test_probes_are_not_made_when_enough_natural_unsupported_claims_exist():
    llm = CountingJudge()
    units, calls = probe_units(many_records(), llm, CHUNK, need_unsupported=0)
    assert units == [] and calls == [] and llm.models == []


def test_probes_use_the_run_judge_and_skip_labelled_claims():
    llm = CountingJudge()
    units, _ = probe_units(
        many_records(),
        llm,
        CHUNK,
        need_unsupported=2,
        judge_model="some-judge",
        exclude_claims={_key("Old claim")},
    )
    assert set(llm.models) == {"some-judge"} and len(llm.models) == 2
    assert {u["claim"] for u in units} == {"new claim"}


def test_k8_row_reuses_the_best_row():
    assert (
        replace(CONFIGS["hybrid_rerank"], k=8, name="hybrid_rerank_k8")
        == (CONFIGS["hybrid_rerank_k8"])
    )
    assert claim(True)["supported_by_context"]


# --- comparison partners, guards and post-hoc labels ---


def test_k8_and_maxp_rows_are_compared_with_their_own_counterparts():
    rerank = CONFIGS["hybrid_rerank"]
    maxp = replace(rerank, name="hybrid_rerank_maxp", rerank_window=320, rerank_overlap=96)
    k8 = replace(maxp, name="hybrid_rerank_maxp_k8", k=8)
    runs = [
        (CONFIGS["baseline"], "a", []),
        (CONFIGS["hybrid"], "b", []),
        (rerank, "c", []),
        (CONFIGS["hybrid_rerank_headers"], "d", []),
        (maxp, "e", []),
        (k8, "f", []),
    ]
    assert [compare.comparison_partner(i, runs) for i in range(1, 6)] == [0, 1, 2, 2, 4]


def test_report_has_guard_metrics_and_labels_the_post_hoc_row():
    rerank = CONFIGS["hybrid_rerank"]
    maxp = replace(rerank, name="hybrid_rerank_maxp", rerank_window=320, rerank_overlap=96)
    rows_a = records(abstain_every=0) + [record("u1", answerable=False, abstained=True)]
    rows_b = records(abstain_every=0) + [record("u1", answerable=False, abstained=False)]
    runs = [(rerank, "c", rows_a), (maxp, "e", rows_b)]
    md, csv_rows = ablation_markdown(runs, "dev", GIT, GOLDEN, "c" * 64, "t")
    assert "| `hybrid_rerank_maxp` vs `hybrid_rerank` | abstention_recall | -1.000 |" in md
    assert "| `hybrid_rerank_maxp` vs `hybrid_rerank` | citation_accuracy |" in md
    assert "post-hoc fix, decided after seeing dev reranker truncation" in md
    assert "MaxP: 320-token windows, overlap 96" in md
    assert "whole passage (truncated at 512)" in md
    assert any(r["metric"] == "diff_vs:hybrid_rerank:abstention_recall" for r in csv_rows)


# --- best row (decides the k = 8 run) ---


def scored(correct_every=1, abstain_every=0, n=20):
    """Rows where every `correct_every`-th answer is correct and the rest incorrect."""
    out = records(n, abstain_every=abstain_every)
    for i, r in enumerate(out):
        if r["judge"] and i % correct_every:
            r["judge"]["correctness"] = "incorrect"
    return out


def test_best_row_takes_correctness_then_lower_false_abstention():
    a = CONFIGS["hybrid"]
    b = CONFIGS["hybrid_rerank"]
    c = CONFIGS["hybrid_rerank_headers"]
    k8 = CONFIGS["hybrid_rerank_k8"]
    # b and c tie on correctness (0.5); c abstains less, so c wins. a is lower (0.25).
    rows_a = scored(correct_every=4)
    rows_b = records(20, abstain_every=2)  # 10 abstained, 10 correct -> 0.5
    rows_c = scored(correct_every=2)  # 10 correct, 10 incorrect, 0 abstained -> 0.5
    runs = [(a, "a", rows_a), (b, "b", rows_b), (c, "c", rows_c), (k8, "d", records())]
    chosen, table = compare.best_row(runs)
    assert [t["config"] for t in table] == [a.name, b.name, c.name]  # k = 8 row not a candidate
    assert table[1]["answer_correctness"] == table[2]["answer_correctness"] == 0.5
    assert chosen == 2  # tie on correctness; c has the lower false-abstention rate
    md, _ = ablation_markdown(runs, "dev", GIT, GOLDEN, "c" * 64, "t")
    assert "## Choice of the k = 8 row" in md
    assert "`hybrid_rerank_k8` is **NOT** the chosen row at k = 8." in md
    runs[3] = (CONFIGS["hybrid_rerank_headers_k8"], "d", records())
    md, _ = ablation_markdown(runs, "dev", GIT, GOLDEN, "c" * 64, "t")
    assert "`hybrid_rerank_headers_k8` is the chosen row at k = 8." in md


def test_summary_compare_is_labelled_and_paired():
    from medquery.eval.compare import summary_compare_markdown

    md = summary_compare_markdown(
        (CONFIGS["baseline_nvidia"], "a", records(correct=False)),
        (CONFIGS["hybrid_headers_tables_answer_v2"], "b", records(correct=True)),
        "test",
        GIT,
        "t",
    )
    assert "computed after the final runs; no tuning" in md
    assert "| answer_correctness | 0.000 | 1.000 | +1.000 |" in md and "**yes**" in md
    assert "hallucination_rate" in md and "false_abstention_rate" in md
