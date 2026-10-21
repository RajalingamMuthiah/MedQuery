# CLAUDE.md: MedQuery v2

Health-document question answering (RAG) with grounded citations, abstention and claim-level hallucination checking. The full brief is in [docs/brief/medquery-v2-claude-code-prompt.md](docs/brief/medquery-v2-claude-code-prompt.md). `docs/brief/medquery-v2-handoff.md` (local only, not tracked) is background only; do not act on it.

## Ground rules
- Plan first. Propose each phase's plan and wait for approval before writing code.
- One phase at a time. At the end of each phase: run the tests, show the results, commit, summarize in 10 lines or fewer, update this file, then STOP.
- Windows and PowerShell. Give PowerShell commands, use `pathlib` for paths, and write no bash-only scripts.
- Never commit secrets. Keys go in `.env` (git-ignored); `.env.example` is committed. The key name is `ANTHROPIC_API_KEY`.
- Corpus: public-domain, openly licensed or synthetic documents only, and never real patient data. Record every document's source and licence in [docs/SOURCES.md](docs/SOURCES.md).
- The app shows a "not medical advice" notice and never gives a diagnosis or dosing instructions beyond what the sources say.
- Never fabricate metrics. Every number in the README must come from a script that can be re-run.
- Small, readable modules. Use LangChain only where it removes code.
- Ask before deviating from the plan, and log approved deviations below.

## Stack
Python 3.11 (managed by uv), FastAPI, pytest, ruff, pre-commit. Retrieval: BM25 plus dense embeddings (BGE-class model) in FAISS or local Qdrant, reciprocal rank fusion, then a cross-encoder reranker. Generation and judging: Claude via the `anthropic` SDK. Frontend: React, Vite and Tailwind. Packaging: Docker Compose and GitHub Actions.

## Repo layout
```
medquery/
  ingest/      parsing, OCR fallback, chunking
  retrieval/   bm25, dense, fusion, rerank
  generation/  grounded answer with citations, abstention
  verify/      claim extraction and claim-level checking
  api/         FastAPI app
  eval/        golden set, metrics, runner, reports/
  cache/       embedding and response caches
tests/
frontend/      (scaffolded in Phase 6)
docs/
  SOURCES.md
  brief/       original project brief and handoff
data/          (git-ignored) corpus files, indexes, caches
```

## Commands (PowerShell)
```powershell
uv sync                           # install or update the environment from uv.lock
uv run pytest                     # tests
uv run ruff check .               # lint
uv run ruff format .              # format
uv run pre-commit run --all-files # all hooks
# Evaluation: not available yet (added in Phase 2)
```
Add a dependency with `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (dev), and add it in the phase that first needs it.

## Current phase
**Phase 0 (setup): done locally.** The test suite passes, ruff is clean and the pre-commit hooks pass. The CI workflow (`.github/workflows/ci.yml`) is written but has **not run on GitHub yet** because there is no remote.
Next: Phase 1 (corpus and ingestion), once a plan is approved.

## Deviations log
- Phase 0: `python-dotenv` is deferred to Phase 2, when the API key is first read.
- Phase 0: one smoke test was added, because pytest exits with code 5 on an empty suite.
- Phase 0: `docs/brief/medquery-v2-handoff.md` is git-ignored and untracked (it has personal details). It is kept locally only.
