"""Live latency benchmark (local only; real Groq key; not run by CI or pytest):
uv run python -m medquery.api.bench

Runs 10 fixed DEV questions through the live app on Groq, with the LLM disk cache and the answer
cache disabled so every call is real, and checks each answered one with /verify. Writes
medquery/eval/reports/<date>_live_latency_<hash>.md from a clean commit. Uses about 18 of the
daily cap and refuses to start without that much left. The test split is never used."""

import hashlib
import json
import sys
import time
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from medquery.api.answer_cache import AnswerCache
from medquery.api.app import create_app, sample_questions
from medquery.api.budget import ANSWER_EXPECTED_TOKENS, VERIFY_EXPECTED_TOKENS
from medquery.api.smoke import wait_ready
from medquery.eval.freeze import load_golden
from medquery.eval.gitinfo import git_state
from medquery.eval.metrics import percentiles
from medquery.eval.report import REPORTS_DIR
from medquery.fileio import atomic_write

N_QUESTIONS = 10
CAP_NEEDED = 18  # 10 asks + up to 8 verifies


def bench_questions(items: list[dict]) -> list[dict]:
    """The 8 /samples questions plus the next answerable dev questions by id (dev only)."""
    chosen = sample_questions(items)
    ids = {s["id"] for s in chosen}
    dev = sorted(
        (i for i in items if i["split"] == "dev" and i["answerable"]), key=lambda i: i["id"]
    )
    for item in dev:
        if len(chosen) >= N_QUESTIONS:
            break
        if item["id"] not in ids:
            chosen.append({"id": item["id"], "question": item["question"], "topic": item["topic"]})
    assert all(i["split"] == "dev" for i in items if i["id"] in {c["id"] for c in chosen})
    return chosen


def stream_events(client: TestClient, question: str) -> tuple[list[tuple[str, dict]], float, float]:
    """(events, seconds to the first delta, seconds to the final event), measured here."""
    started = time.perf_counter()
    first_delta = done = None
    events, buffer = [], ""
    with client.stream("POST", "/ask/stream", json={"question": question}) as response:
        for chunk in response.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
                name, data = fields.get("event", ""), json.loads(fields.get("data", "{}"))
                now = time.perf_counter() - started
                if name == "delta" and first_delta is None:
                    first_delta = now
                if name in ("final", "error"):
                    done = now
                events.append((name, data))
    return events, first_delta, done


def p(values: list[float]) -> str:
    q = percentiles([v for v in values if v is not None])
    if q["p50"] is None:
        return "n/a | n/a"
    return f"{q['p50']:.2f} | {q['p95']:.2f}"


def main() -> int:
    git = git_state()
    if git["dirty"] is not False:
        print("refused: the benchmark report needs a clean, committed git tree")
        return 1
    from medquery.llm import get_client

    items, _ = load_golden("v1")
    questions = bench_questions(items)
    app = create_app(
        llm=get_client(cache=False),
        answer_cache=AnswerCache(None, enabled=False),
        rate_per_min=1000,  # local measurement; the daily cap still applies
    )
    rows, verify_rows = [], []
    with TestClient(app) as client:
        health = wait_ready(client)
        if not health["ready"]:
            print(f"not ready: {health['status']} {health.get('error')}")
            return 1
        if health["daily_remaining"] < CAP_NEEDED:
            print(f"refused: {health['daily_remaining']} of the daily cap left; need {CAP_NEEDED}")
            return 1
        for q in questions:
            events, first_delta, done = stream_events(client, q["question"])
            name, final = events[-1]
            if name != "final":
                print(f"{q['id']}: {name} {final}")
                return 1
            lat = final["latency"]
            rows.append(
                {
                    "id": q["id"],
                    "abstained": final["abstained"],
                    "retrieval_s": lat["retrieval_ms"] / 1000,
                    "model_ttft_s": lat["ttft_s"],
                    "generation_s": lat["generation_s"],
                    "client_ttft_s": first_delta,
                    "client_total_s": done,
                }
            )
            print(
                f"{q['id']}: {'abstained' if final['abstained'] else 'answered'}, "
                f"first text {first_delta}, done {done:.2f} s"
            )
            if final["abstained"]:
                continue
            body = {
                "question": q["question"],
                "answer": final["answer"],
                "passage_ids": [x["id"] for x in final["passages"]],
            }
            estimate = client.post("/verify/estimate", json=body).json()["wait_s"]
            started = time.perf_counter()
            v = client.post("/verify", json=body).json()
            verify_rows.append(
                {
                    "id": q["id"],
                    "estimate_s": estimate,
                    "wait_s": v.get("wait_s"),
                    "total_s": time.perf_counter() - started,
                    "verified": v.get("verified"),
                }
            )
            print(
                f"  verify: {v.get('verified')}, wait {v.get('wait_s')} s "
                f"(estimated {estimate}), total {verify_rows[-1]['total_s']:.2f} s"
            )

    created = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    key = hashlib.sha256(f"{created}|{[q['id'] for q in questions]}".encode()).hexdigest()[:10]
    path = REPORTS_DIR / f"{created[:10]}_live_latency_{key}.md"
    answered = [r for r in rows if not r["abstained"]]
    lines = [
        "# MedQuery live latency (Groq, dev questions)",
        "",
        f"- Created (UTC): {created}",
        f"- Git commit: `{git['commit']}` (clean working tree)",
        f"- Live config: {health['config']['answer_prompt']}, corpus {health['config']['corpus']}, "
        f"generator `{health['config']['generator']}` on {health['config']['provider']}, "
        f"verifier `{health['config']['verifier']}`",
        f"- Questions: {len(rows)} fixed dev questions ({len(answered)} answered, "
        f"{len(rows) - len(answered)} abstained); ids {', '.join(r['id'] for r in rows)}. "
        "The test split is never used.",
        "- Every call is real: the LLM disk cache and the answer cache were disabled.",
        f"- Throttle bookings: answer {ANSWER_EXPECTED_TOKENS} and verify {VERIFY_EXPECTED_TOKENS} "
        "expected output tokens (medquery/api/budget.py).",
        "- Measured through the app in-process (FastAPI TestClient), so network time to a browser "
        "is not included.",
        "",
        f"## Answers (n = {len(rows)})",
        "",
        "| stage | p50 (s) | p95 (s) |",
        "|---|---|---|",
        f"| retrieval (server) | {p([r['retrieval_s'] for r in rows])} |",
        f"| time to first token: model call (server) | {p([r['model_ttft_s'] for r in rows])} |",
        f"| time to first token: whole request | {p([r['client_ttft_s'] for r in rows])} |",
        f"| full generation (model call) | {p([r['generation_s'] for r in rows])} |",
        f"| whole request to the final event | {p([r['client_total_s'] for r in rows])} |",
        "",
        f"## Verify (n = {len(verify_rows)})",
        "",
        "| stage | p50 (s) | p95 (s) |",
        "|---|---|---|",
        f"| throttle wait, estimated | {p([v['estimate_s'] for v in verify_rows])} |",
        f"| throttle wait, actual | {p([v['wait_s'] for v in verify_rows])} |",
        f"| whole /verify request | {p([v['total_s'] for v in verify_rows])} |",
        "",
        f"Verifications completed: {sum(bool(v['verified']) for v in verify_rows)} of "
        f"{len(verify_rows)} (the rest reported 'unverified').",
        "",
        "With n = 10 the p95 is close to the slowest call; treat it as indicative.",
    ]
    with atomic_write(path) as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
