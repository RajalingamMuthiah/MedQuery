"""Phase 4 provider switch (NVIDIA NIM) and the judge budget. Offline: fake clients only."""

import hashlib
import json
from dataclasses import asdict

import pytest

from medquery import config
from medquery.eval import budget
from medquery.eval.runner import CONFIGS, RETRIEVAL_DEFAULTS, RunConfig, evaluate_item, run_id
from medquery.generation.answer import answer_from_hits
from medquery.llm import LLMError, Router, get_client
from medquery.llm.base import Completion
from tests.test_rag import CHUNKS


class RecordingClient:
    def __init__(self, name):
        self.name, self.calls = name, []

    def complete(self, messages, **kw):
        self.calls.append(kw)
        return Completion(text=f"from {self.name}", model=kw["model"], finish_reason="stop")


def router(*providers):
    clients = {p: RecordingClient(p) for p in providers}
    return Router(clients), clients


def test_explicit_provider_overrides_the_model_default():
    r, clients = router("groq", "nvidia")
    assert r.complete([], model="openai/gpt-oss-20b").text == "from groq"  # MODEL_PROVIDER
    assert r.complete([], model="openai/gpt-oss-20b", provider="nvidia").text == "from nvidia"
    assert "provider" not in clients["nvidia"].calls[0]  # consumed by the router


def test_missing_nvidia_key_names_the_variable():
    r, _ = router("groq")
    with pytest.raises(LLMError, match="NVIDIA_API_KEY is not set"):
        r.complete([], model="openai/gpt-oss-20b", provider="nvidia")


def test_get_client_builds_nvidia_and_redacts_every_key(monkeypatch):
    monkeypatch.setattr("medquery.llm.load_dotenv", lambda *a, **k: None)
    keys = {"GROQ_API_KEY": "gk-1", "OPENAI_API_KEY": "ok-2", "NVIDIA_API_KEY": "nk-3"}
    for name, value in keys.items():
        monkeypatch.setenv(name, value)
    client = get_client()
    assert set(client._clients) == {"groq", "openai", "nvidia"}
    nvidia = client._clients["nvidia"]
    assert set(nvidia._secrets) == set(keys.values())
    assert "nk-3" not in repr(client) and "nk-3" not in repr(nvidia)
    assert nvidia._redact("bad key nk-3 and gk-1") == "bad key *** and ***"


def test_nvidia_price_is_zero_and_groq_keeps_its_list_price():
    assert config.cost_usd("openai/gpt-oss-20b", 1_000_000, 1_000_000, "nvidia") == 0.0
    assert config.cost_usd("openai/gpt-oss-20b", 1_000_000, 0) == pytest.approx(0.075)
    assert config.cost_usd("openai/gpt-oss-20b", 1_000_000, 0, "groq") == pytest.approx(0.075)
    assert "nvidia" in config.price_note()
    assert config.PROVIDERS["nvidia"]["key_env"] == "NVIDIA_API_KEY"


def test_answer_passes_the_provider_only_when_set():
    r, clients = router("groq", "nvidia")
    answer_from_hits("q", [], 0.0, r)
    answer_from_hits("q", [], 0.0, r, provider="nvidia")
    assert len(clients["groq"].calls) == 1 and len(clients["nvidia"].calls) == 1
    assert clients["nvidia"].calls[0]["reasoning_effort"] == "low"  # same settings as Groq


def test_earlier_run_ids_are_unchanged_and_nvidia_runs_get_their_own():
    assert RETRIEVAL_DEFAULTS["generator_provider"] == "groq"
    for name in ("baseline", "hybrid", "hybrid_rerank_headers_maxp"):
        fields = {k: v for k, v in asdict(CONFIGS[name]).items() if k != "generator_provider"}
        fields = {
            k: v
            for k, v in fields.items()
            if k not in RETRIEVAL_DEFAULTS or v != RETRIEVAL_DEFAULTS[k]
        }
        payload = {**fields, "split": "dev", "golden": "g", "chunks": "c"}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        assert run_id(CONFIGS[name], "dev", "g", "c") == expected
    assert run_id(CONFIGS["hybrid_nvidia"], "dev", "g", "c") != run_id(
        CONFIGS["hybrid"], "dev", "g", "c"
    )
    nvidia = CONFIGS["hybrid_rerank_headers_maxp_nvidia"]
    assert asdict(nvidia) | {"name": "x", "generator_provider": "groq"} == asdict(
        CONFIGS["hybrid_rerank_headers_maxp"]
    ) | {"name": "x"}


class OneHitRetriever:
    def retrieve(self, question, k):
        from medquery.retrieval.dense import Hit

        return [Hit(CHUNKS[0], 1.0, 1)], 0.01


class NullMapper:
    def map(self, evidence):
        from types import SimpleNamespace

        return SimpleNamespace(chunk_ids=(), status="missing")


def test_evaluate_item_routes_generation_to_nvidia_and_costs_it_at_zero():
    r, clients = router("groq", "nvidia", "openai")
    item = {
        "id": "g1",
        "split": "dev",
        "topic": "tb",
        "type": "factual",
        "answerable": True,
        "question": "q?",
        "reference_answer": "r",
        "evidence": [],
    }
    rec = evaluate_item(item, OneHitRetriever(), r, NullMapper(), CONFIGS["hybrid_nvidia"])
    assert clients["nvidia"].calls and not clients["groq"].calls
    assert rec["cost_usd"]["generation"] == 0.0
    rec = evaluate_item(item, OneHitRetriever(), r, NullMapper(), RunConfig(name="b"))
    assert len(clients["groq"].calls) == 1


# --- judge budget ---


def write_run(runs_dir, rid, provider, judge_costs):
    d = runs_dir / rid
    d.mkdir(parents=True)
    cfg = {"name": rid, "generator_provider": provider} if provider else {"name": rid}
    (d / "run.json").write_text(json.dumps({"config": cfg, "split": "dev"}), encoding="utf-8")
    lines = [
        json.dumps({"id": f"g{i}", "judge": {"x": 1}, "cost_usd": {"generation": 0, "judge": c}})
        for i, c in enumerate(judge_costs)
    ]
    (d / "items.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_budget_counts_only_phase4_runs_and_refuses_over_the_cap(tmp_path):
    write_run(tmp_path, "old", None, [0.01] * 40)  # Phase 3 (Groq): not counted as spent
    write_run(tmp_path, "p4a", "nvidia", [0.01] * 30)
    assert budget.spent(tmp_path) == pytest.approx(0.30)
    ok = budget.check(tmp_path, "new", todo=44)
    assert ok.ok and ok.estimate == pytest.approx(44 * 0.01 * 1.5)
    tight = budget.check(tmp_path, "new", todo=44, cap=0.5)
    assert not tight.ok and "cap $0.50" in tight.line()
    resumed = budget.check(tmp_path, "p4a", todo=14)  # own finished items count as spent
    assert resumed.spent == pytest.approx(0.30)


def test_run_refuses_when_the_budget_would_be_exceeded(monkeypatch, capsys):
    import medquery.eval.__main__ as cli

    monkeypatch.setattr(
        cli.budget, "check", lambda *a, **k: budget.BudgetCheck(2.9, 0.5, budget.PHASE4_CAP_USD)
    )
    monkeypatch.setattr(cli, "sha256_file", lambda path: "x")  # no corpus needed (CI has none)
    monkeypatch.setattr(cli, "chunks_path", lambda corpus: cli.paths.ROOT / "pyproject.toml")
    assert cli.main(["run", "--split", "dev", "--config", "hybrid_nvidia"]) == 1
    out = capsys.readouterr().out
    assert "refused" in out and "$2.90 spent" in out


# --- provider-check report ---


def test_provider_check_flags_real_differences_and_reparses_citations():
    from medquery.eval.compare import provider_check_markdown, reparse_citations
    from tests.test_compare import GIT, GOLDEN, records

    groq = records(correct=True)
    nvidia_same = [dict(r) for r in groq]
    nvidia_worse = records(correct=False)
    pair = (CONFIGS["hybrid"], "g1", groq, CONFIGS["hybrid_nvidia"], "n1", nvidia_same)
    md, rows, any_real = provider_check_markdown([pair], GIT, GOLDEN, "c" * 64, "t")
    assert not any_real and "provider switch is neutral" in md
    assert {r["metric"] for r in rows} >= {"answer_correctness", "citation_validity"}
    worse = (CONFIGS["hybrid"], "g1", groq, CONFIGS["hybrid_nvidia"], "n2", nvidia_worse)
    md, _, any_real = provider_check_markdown([worse], GIT, GOLDEN, "c" * 64, "t")
    assert any_real and "**yes**" in md
    rec = groq[0] | {"answer": "x [dm‑stats‑2020:0014]", "cited_ids": ["2020:0014"]}
    assert reparse_citations([rec])[0]["cited_ids"] == ["dm-stats-2020:0014"]


def test_provider_check_cli_needs_a_clean_tree(monkeypatch, capsys):
    import medquery.eval.__main__ as cli

    monkeypatch.setattr(cli, "git_state", lambda: {"commit": "x", "dirty": True})
    assert cli.main(["provider-check", "--configs", "hybrid"]) == 1
    assert "clean, committed" in capsys.readouterr().out


def test_provider_check_section_reports_real_differences_and_primary_metrics(tmp_path):
    from medquery.eval.compare import provider_check_section

    path = tmp_path / "pc.csv"
    path.write_text(
        "config,metric,groq,nvidia,diff,ci_low,ci_high\n"
        "hybrid,answer_correctness,0.9,0.9,0.0,0.0,0.0\n"
        "hybrid,faithfulness,0.91,0.93,0.016,0.0003,0.036\n"
        "hybrid,hallucination_rate,0.1,0.08,-0.02,-0.05,0.01\n",
        encoding="utf-8",
    )
    text = "\n".join(provider_check_section(path))
    assert "3 paired tests" in text and "faithfulness +0.016" in text
    assert "(favours NVIDIA)" in text and "unchanged on every row" in text
