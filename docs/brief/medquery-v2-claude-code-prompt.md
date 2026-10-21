# MedQuery v2: Claude Code prompt

Paste everything below the line into Claude Code, starting in plan mode, in an empty project folder.

---

You are helping me rebuild MedQuery v2, a health-document question-answering system (RAG) with a built-in hallucination check. I am an AI/ML engineer. The goal is a portfolio-quality project whose accuracy I can measure and show honestly.

## Ground rules
- Start in plan mode. Read the folder, propose the plan, and wait for my approval before writing code for each phase.
- Work one phase at a time. At the end of each phase: run the tests, show the results, make a git commit, summarize in 10 lines or fewer, then STOP and wait for me.
- I use Windows and PowerShell. Give commands for PowerShell, use pathlib for paths, and do not write bash-only scripts.
- Never commit secrets. Use a .env file and a committed .env.example. The key is ANTHROPIC_API_KEY.
- Use only public-domain or openly licensed medical documents, or synthetic ones. Never use real patient data. Record the source and licence of every document in docs/SOURCES.md.
- The app must show a clear "not medical advice" notice and must never give a diagnosis or dosing instructions beyond what the source documents say.
- Never fabricate metrics. Every number in the README must come from a script I can re-run.
- Prefer small, readable modules. Use LangChain only where it removes code.
- If you want to deviate from this plan, tell me why and ask first.

## First task: create CLAUDE.md
Write a CLAUDE.md that summarizes these rules, the repo layout below, the commands to run tests and the evaluation, and the current phase. Keep it updated at the end of each phase.

## Stack
Python 3.11, FastAPI, pytest, ruff. Retrieval: BM25 plus dense embeddings (a small open embedding model, for example BGE) in FAISS or Qdrant local, merged with reciprocal rank fusion, then a cross-encoder reranker. Generation and judging: Anthropic Claude via the official SDK. Frontend: React with Vite and Tailwind. Packaging: Docker Compose and GitHub Actions.

## Repo layout
medquery/
  ingest/      parsing, OCR fallback, chunking
  retrieval/   bm25, dense, fusion, rerank
  generation/  grounded answer with citations, abstention
  verify/      claim extraction and claim-level checking
  api/         FastAPI app
  eval/        golden set, metrics, runner, reports
  cache/       embedding and response caches
tests/
frontend/
docs/
  SOURCES.md

## Phases
**Phase 0, setup:** repo, CLAUDE.md, pyproject, ruff, pytest, pre-commit, .env.example, folder structure.
Done when: the empty test suite passes and CI runs.

**Phase 1, corpus and ingestion:** collect 10 to 20 open-licence health documents (for example WHO or CDC fact sheets, open-access guidelines and papers). Parse PDFs while keeping page numbers, section headings and tables. Use an OCR fallback (PaddleOCR or Tesseract) for scanned pages. Chunk by section with metadata (document, page, section).
Done when: a script ingests the corpus, and tests cover a table page and a scanned page.

**Phase 2, baseline and evaluation first:** build the simplest possible RAG (dense retrieval and one prompt). Build the evaluation harness before improving anything.
- Golden set: 60 to 100 questions generated from the corpus with an LLM, then reviewed by me (at least 25 percent checked by hand). About 20 percent must be unanswerable from the documents.
- Metrics: faithfulness, answer relevancy, context precision, context recall, citation accuracy, abstention accuracy, claim-level hallucination rate, latency p50 and p95, and cost per query.
- Output: a results table written to eval/reports/ (CSV and Markdown) with the run date and config.
Done when: one command produces the baseline report.

**Phase 3, retrieval upgrades:** add BM25, rank fusion, then the reranker. Run the evaluation after each addition.
Done when: the report shows each step's effect (the ablation table).

**Phase 4, grounded generation:** answers must cite passages (document, page and the supporting quote). If the evidence is weak, abstain with a short explanation.
Done when: citation accuracy and abstention accuracy are measured and reported.

**Phase 5, claim-level verification:** split each answer into atomic claims, check each one against the retrieved text, and remove or repair unsupported claims (at most one retry). Return a per-claim support status.
Done when: hallucination rate with and without verification is reported, along with the extra latency and cost.

**Phase 6, API and UI:** FastAPI endpoints for upload, ask and evaluate. The React UI streams answers, highlights the source passage for each claim, and shows a confidence and abstention state. Show the "not medical advice" notice.
Done when: I can upload a document, ask a question, and see cited, verified answers.

**Phase 7, performance:** async calls, embedding cache, response cache, prompt caching, and batching where it helps. Measure before and after.
Done when: the report shows the latency and cost change.

**Phase 8, packaging:** Docker Compose, README with the final results table and an architecture diagram, a short demo script, and GitHub Actions that run tests and a small evaluation subset.
Done when: a fresh clone runs with two commands.

**Optional Phase 9, multilingual:** accept questions in Hindi or Marathi and answer from English documents, with evaluation on a small translated question set.

## Final README must include
The ablation table (baseline, plus hybrid, plus reranker, plus verification), the metric definitions, known limitations, and how to reproduce every number.

Begin with plan mode: read the folder, ask me any questions you need (maximum five), then propose the Phase 0 plan.
