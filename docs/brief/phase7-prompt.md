Phase 7: speed and polish of the live app. Plan only; no code until I approve. No changes to evaluation code, golden sets, frozen labels or committed reports; the test split is never used.

1. Verify wait: the client throttle reserves full max_tokens (4000 for verify_v2) against Groq's 8K TPM. Reserve expected output tokens instead (justify the number from recorded completion tokens, e.g. p95 + margin), keep a safe fallback if Groq returns 429, and show an estimated wait in the UI when the throttle must wait.
2. Streaming: stream /ask answers to the frontend (SSE or fetch streaming), with citations and sources finalised when the stream ends; keep the non-streaming endpoint working for tests and the smoke script. Abstention still renders as the "not covered" card.
3. Answer cache: identical (normalised) questions return the stored response instantly without calling Groq or counting toward the daily cap; show "cached" subtly in the UI. Bounded size, persisted in data/.
4. Warm start: load the corpus, embeddings and model at startup; /health reports readiness; the frontend shows "starting up" until ready.
5. Live latency benchmark: a script that runs ~10 DEV questions through the live app on Groq (respecting the daily cap) and writes a committed report with p50/p95 for retrieval, time-to-first-token, full generation, and verify. README "Live performance" section from that report only.
Tests offline for each part (fake LLM/clock); frontend tests for streaming and cached states. CLAUDE.md, commits, push, CI green, summary, STOP.

Ask me up to 3 questions if anything conflicts with the repo.
