# MedQuery live latency (Groq, dev questions): paced (realistic use)

- Mode: paced (realistic use): one request at most every 30 s
- Created (UTC): 2026-10-26 06:16
- Git commit: `fd23ea0a05c6f418ac56e88e8a7a0824e45f575f` (clean working tree)
- Live config: answer_v2:0180eca9e1d7, corpus v2, generator `openai/gpt-oss-20b` on groq, verifier `verify_v2:0323a83893fc`
- Questions: 10 fixed dev questions (7 answered, 3 abstained); ids g001, g007, g031, g032, g064, g065, g023, g024, g009, g011. The test split is never used.
- Every call is real: the LLM disk cache and the answer cache were disabled.
- Throttle bookings: answer 400 and verify 600 expected output tokens (medquery/api/budget.py).
- Measured through the app in-process (FastAPI TestClient), so network time to a browser is not included.

## Answers (n = 10)

| stage | p50 (s) | p95 (s) |
|---|---|---|
| retrieval (server) | 0.02 | 0.03 |
| time to first token: model call (server) | 0.77 | 1.18 |
| time to first token: whole request | 1.20 | 46.38 |
| full generation (model call) | 0.83 | 1.22 |
| whole request to the final event | 1.25 | 46.43 |

## Verify (n = 7)

| stage | p50 (s) | p95 (s) |
|---|---|---|
| throttle wait, estimated | 0.00 | 21.00 |
| throttle wait, actual | 0.00 | 21.00 |
| whole /verify request | 0.88 | 22.13 |

Verifications completed: 7 of 7 (the rest reported 'unverified').

With n = 10 the p95 is close to the slowest call; treat it as indicative.
