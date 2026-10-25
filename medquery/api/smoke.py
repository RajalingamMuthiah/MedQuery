"""End-to-end smoke test against the real Groq key (local only; not run by CI or pytest):
uv run python -m medquery.api.smoke

Asks two dev questions and one dev unanswerable question through the real app, then checks the
first answer with /verify. Counts toward the daily cap like any other request."""

import sys
import time

from fastapi.testclient import TestClient

from medquery.api.app import create_app, sample_questions
from medquery.eval.freeze import load_golden


def main() -> int:
    items, _ = load_golden("v1")
    samples = sample_questions(items)
    picks = samples[:2] + samples[-1:]  # two answerable dev questions, one unanswerable
    with TestClient(create_app()) as client:
        return run(client, picks)


def wait_ready(client: TestClient, timeout_s: float = 300) -> dict:
    """Poll /health until the background warm start has finished (or failed)."""
    started = time.monotonic()
    while True:
        health = client.get("/health").json()
        if health["ready"] or health["status"] == "error" or time.monotonic() - started > timeout_s:
            return health
        time.sleep(1)


def run(client: TestClient, picks: list[dict]) -> int:
    health = wait_ready(client)
    if not health["ready"]:
        print(f"not ready: {health['status']} {health.get('error')}")
        return 1
    print("health:", health["config"])
    first = None
    for s in picks:
        r = client.post("/ask", json={"question": s["question"]})
        body = r.json()
        if r.status_code != 200:
            print(f"{s['id']}: HTTP {r.status_code} {body.get('error')}")
            return 1
        first = first or (s, body)
        print(
            f"\n{s['id']} ({s['topic']}): {'ABSTAINED' if body['abstained'] else 'answered'}; "
            f"retrieval {body['latency']['retrieval_ms']} ms, generation "
            f"{body['latency']['generation_s']} s\n  {body['answer'][:300]}"
        )
    s, body = first
    if not body["abstained"]:
        r = client.post(
            "/verify",
            json={
                "question": s["question"],
                "answer": body["answer"],
                "passage_ids": [p["id"] for p in body["passages"]],
            },
        )
        v = r.json()
        print(f"\nverify ({v.get('verifier')}): HTTP {r.status_code}, {v.get('latency_s')} s")
        for sentence in v.get("sentences", []):
            print(f"  [{sentence['status']}] {sentence['text'][:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
