"""Phase 6 API (offline): fake retriever and LLM injected through the app factory."""

import json

import pytest
from fastapi.testclient import TestClient

from medquery import config
from medquery.api.app import create_app, sample_questions
from medquery.api.highlight import highlights
from medquery.api.limits import DailyCap, RateLimiter
from medquery.generation.answer import NOT_MEDICAL_ADVICE
from medquery.generation.prompts import ABSTAIN_TEXT
from medquery.llm.base import Completion, LLMError
from medquery.retrieval.dense import Hit
from tests.test_rag import CHUNKS

ANSWER = (
    "Rifampin is given daily for 4 months [tb:0001]. "
    "This guidance is from 2003 and has been superseded [htn:0001]."
)
TB = CHUNKS[0] | {"id": "tb:0001", "text": "Rifampin is given daily for 4 months.\nOther text."}
HTN = CHUNKS[2] | {"id": "htn:0001"}
PASSAGES = {"tb:0001": TB, "htn:0001": HTN}
GOLDEN = [
    {"id": "g001", "split": "dev", "topic": "tb", "answerable": True, "question": "dev tb 1?"},
    {"id": "g002", "split": "dev", "topic": "tb", "answerable": True, "question": "dev tb 2?"},
    {"id": "g003", "split": "dev", "topic": "tb", "answerable": True, "question": "dev tb 3?"},
    {"id": "g004", "split": "dev", "topic": "dm", "answerable": False, "question": "dev un?"},
    {"id": "g005", "split": "test", "topic": "tb", "answerable": True, "question": "TEST tb?"},
    {"id": "g006", "split": "test", "topic": "dm", "answerable": False, "question": "TEST un?"},
]


class FakeRetriever:
    def retrieve(self, question, k):
        return [Hit(TB, 1.0, 1), Hit(HTN, 0.5, 2)], 0.012


class FakeLLM:
    """Generation returns `answer`; verification returns `verify_reply` (JSON or text)."""

    def __init__(self, answer=ANSWER, verify_reply=None, cached=False, fail=None):
        self.answer, self.verify_reply, self.cached, self.fail = answer, verify_reply, cached, fail
        self.calls = []

    def complete(self, messages, **kw):
        self.calls.append(kw)
        if self.fail:
            raise self.fail
        if "check an answer against source passages" in messages[0]["content"]:
            reply = self.verify_reply
            text = reply if isinstance(reply, str) else json.dumps(reply)
        else:
            text = self.answer
        return Completion(text=text, model=kw["model"], finish_reason="stop", cached=self.cached)


def client(tmp_path, llm=None, **kw):
    app = create_app(
        retriever=FakeRetriever(),
        llm=llm or FakeLLM(),
        chunks=PASSAGES,
        golden=GOLDEN,
        usage_path=tmp_path / "usage.json",
        **{"rate_per_min": 100, "daily_cap": 100, **kw},
    )
    return TestClient(app)


def test_ask_returns_sentences_citations_passages_and_the_notice(tmp_path):
    llm = FakeLLM()
    r = client(tmp_path, llm).post("/ask", json={"question": "  How is rifampin given?  "})
    body = r.json()
    assert r.status_code == 200 and not body["abstained"] and body["notice"] == NOT_MEDICAL_ADVICE
    assert [s["citations"] for s in body["sentences"]] == [["tb:0001"], ["htn:0001"]]
    assert body["sentences"][0]["highlights"]["tb:0001"] == [[0, 37]]
    passages = {p["id"]: p for p in body["passages"]}
    assert passages["htn:0001"]["superseded"] and passages["tb:0001"]["title"] == "Doc"
    assert body["latency"]["retrieval_ms"] == 12.0 and body["question"] == "How is rifampin given?"
    gen = llm.calls[0]
    assert gen["provider"] == "groq" and gen["model"] == config.GENERATOR_MODEL


def test_abstention_is_an_answer_not_an_error(tmp_path):
    r = client(tmp_path, FakeLLM(answer=ABSTAIN_TEXT)).post("/ask", json={"question": "Mars?"})
    assert r.status_code == 200 and r.json()["abstained"] and r.json()["sentences"] == []


def test_question_length_cap_and_empty_question(tmp_path):
    c = client(tmp_path)
    long = c.post("/ask", json={"question": "x" * 501})
    assert long.status_code == 422 and "limit is 500" in long.json()["error"]
    assert long.json()["notice"] == NOT_MEDICAL_ADVICE
    assert "empty" in c.post("/ask", json={"question": "   "}).json()["error"]


def verify_body(**kw):
    return {"question": "q?", "answer": ANSWER, "passage_ids": ["tb:0001", "htn:0001"]} | kw


GOOD = {
    "sentences": [
        {"n": 1, "kind": "fact", "claims": [{"text": "daily", "supported": True}]},
        {"n": 2, "kind": "note", "claims": []},
    ]
}


def test_verify_returns_a_status_per_sentence(tmp_path):
    llm = FakeLLM(verify_reply=GOOD)
    r = client(tmp_path, llm).post("/verify", json=verify_body())
    body = r.json()
    assert r.status_code == 200 and body["verified"]
    assert [s["status"] for s in body["sentences"]] == ["supported", "note"]
    assert body["verifier"].startswith("verify_v2:") and llm.calls[0]["max_tokens"] == 4000
    assert llm.calls[0]["provider"] == "groq" and llm.calls[0]["model"] != config.JUDGE_MODEL
    bad = GOOD | {
        "sentences": [
            GOOD["sentences"][0] | {"claims": [{"text": "x", "supported": False}]},
            GOOD["sentences"][1],
        ]
    }
    r = client(tmp_path, FakeLLM(verify_reply=bad)).post("/verify", json=verify_body())
    assert [s["status"] for s in r.json()["sentences"]] == ["unsupported", "note"]


def test_failed_or_empty_verification_is_unverified(tmp_path):
    r = client(tmp_path, FakeLLM(verify_reply="")).post("/verify", json=verify_body())
    body = r.json()
    assert r.status_code == 200 and not body["verified"]
    assert [s["status"] for s in body["sentences"]] == ["unverified", "unverified"]


def test_verify_rejects_unknown_passages(tmp_path):
    r = client(tmp_path).post("/verify", json=verify_body(passage_ids=["nope:0001"]))
    assert r.status_code == 422 and "unknown passage ids" in r.json()["error"]


def test_per_ip_rate_limit(tmp_path):
    c = client(tmp_path, rate_per_min=2)
    codes = [c.post("/ask", json={"question": "q?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_daily_cap_counts_real_calls_only_and_survives_a_restart(tmp_path):
    c = client(tmp_path, daily_cap=2)
    assert [c.post("/ask", json={"question": "q?"}).status_code for _ in range(2)] == [200, 200]
    restarted = client(tmp_path, daily_cap=2)
    r = restarted.post("/ask", json={"question": "q?"})
    assert r.status_code == 429 and "daily request limit" in r.json()["error"]
    cached = client(tmp_path / "other", FakeLLM(cached=True), daily_cap=1)
    assert [cached.post("/ask", json={"question": "q?"}).status_code for _ in range(3)] == [200] * 3


def test_cap_resets_on_a_new_utc_day(tmp_path):
    day = ["2026-10-25"]
    cap = DailyCap(tmp_path / "u.json", 1, today=lambda: day[0])
    cap.record()
    assert cap.remaining() == 0
    day[0] = "2026-10-26"
    assert cap.remaining() == 1


def test_rate_limiter_window():
    t = [0.0]
    lim = RateLimiter(1, clock=lambda: t[0])
    assert lim.allow("a") and not lim.allow("a") and lim.allow("b")
    t[0] = 60.0
    assert lim.allow("a")


def test_llm_errors_are_503_without_secrets(tmp_path):
    r = client(tmp_path, FakeLLM(fail=LLMError("upstream said no"))).post(
        "/ask", json={"question": "q?"}
    )
    assert r.status_code == 503 and "could not be reached" in r.json()["error"]


def test_cors_allows_only_the_frontend_origin(tmp_path):
    c = client(tmp_path, cors_origin="http://localhost:5173")
    ok = c.options(
        "/ask", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    )
    bad = c.options(
        "/ask", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"}
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in bad.headers


def test_keys_are_never_returned(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-secret-value")
    body = client(tmp_path).get("/health").json()
    assert body["keys_present"]["groq"] is True and "gsk-secret-value" not in json.dumps(body)


def test_samples_come_from_the_dev_split_only(tmp_path):
    body = client(tmp_path).get("/samples").json()
    ids = {s["id"] for s in body["samples"]}
    assert ids == {"g001", "g002", "g004"}  # 2 answerable per topic + unanswerable, dev only
    assert not any("TEST" in s["question"] for s in body["samples"])
    assert all(s["id"] not in {"g005", "g006"} for s in sample_questions(GOLDEN))


def test_documents_list_the_ten_manifest_documents(tmp_path):
    docs = client(tmp_path).get("/documents").json()["documents"]
    assert len(docs) == 10 and all(d["licence"] for d in docs)
    assert any(d["superseded"] for d in docs)  # JNC 7 Express


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("Rifampin is given daily for 4 months [tb:0001].", [[0, 37]]),
        ("Unrelated words [tb:0001].", []),
    ],
)
def test_highlight_finds_the_matching_passage_segment(sentence, expected):
    assert highlights(sentence, TB["text"]) == expected
