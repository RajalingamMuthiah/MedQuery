Phase 6: API and web interface for the live config. Plan only; no code until I approve. No changes to evaluation code, golden sets, frozen labels or committed reports.

## 1. Backend (FastAPI)
- POST /ask: the live config (hybrid + headers + corpus v2 + answer_v2, no reranker), generation on Groq. Returns: answer text, abstained flag, sentences with their citations, each citation's passage (doc title, publisher, as-of year, SUPERSEDED flag, page, section, text), retrieval latency and generation latency.
- POST /verify (optional, asynchronous): runs the verifier on a given answer (on Groq) and returns per-sentence verdicts: supported / unsupported / note / unverified (verifier failed). Never blocks /ask. Use verify_v2 (higher max_tokens, detect reasoning-only/empty replies -> "unverified"); version and hash it as a new prompt.
- GET /health, GET /documents (the corpus manifest: titles, publishers, years, licences, superseded flags).
- Safety: question length cap (and a clear error), basic per-IP rate limit, and a daily request cap so the Groq free tier can't be exhausted by one user; keys only from .env, never logged or returned; CORS restricted to the frontend origin; "not medical advice" in every response.
- Uses the existing medquery package (retrieval, prompts, LLM client, cache); no duplicated logic.

## 2. Frontend (React, in frontend/)
- Ask box; answer with inline citation chips; clicking a chip opens the source passage with document, page and section, and highlights the cited text where possible.
- Abstention shown clearly ("The documents don't cover this"), not as an error.
- SUPERSEDED sources visibly flagged (e.g. JNC 7, 2003).
- "Check this answer" button -> calls /verify and colours sentences: green = supported, amber = unverified, red/struck = unsupported, grey = note.
- A "sample questions" list drawn ONLY from the dev split (never test).
- Persistent "Not medical advice" banner; a "Sources" page listing the 10 documents with licences.
- Clean, accessible, works on mobile widths.

## 3. Tests and rules
- Offline API tests with fake LLM/retriever (ask, abstain, verify success/failure -> unverified, length cap, rate limit); a small frontend test; an end-to-end smoke script that runs locally against the real Groq key (not in CI).
- One command each to run backend and frontend locally (documented in README). Docker is Phase 8.
- Record any deviations in CLAUDE.md. Logical commits, push, CI green, summary, STOP.

Ask me up to 3 questions if anything conflicts with the repo.
