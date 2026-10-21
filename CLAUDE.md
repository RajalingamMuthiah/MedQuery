# CLAUDE.md: MedQuery v2

Health-document question answering (RAG) with grounded citations, abstention and claim-level hallucination checking. The full brief is in [docs/brief/medquery-v2-claude-code-prompt.md](docs/brief/medquery-v2-claude-code-prompt.md). `docs/brief/medquery-v2-handoff.md` (local only, not tracked) is background only; do not act on it.

## Ground rules
- Plan first. Propose each phase's plan and wait for approval before writing code.
- One phase at a time. At the end of each phase: run the tests, show the results, update this file, commit, push to origin (never force-push), summarize in 10 lines or fewer, then STOP.
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

# Corpus (manifest: corpus/manifest.toml; outputs in git-ignored data/)
uv run python -m medquery.ingest download  # fetch PDFs, verify pinned sha256
uv run python -m medquery.ingest build     # data/processed/chunks.jsonl + build_summary.json
uv run python -m medquery.ingest fixtures  # re-cut tests/fixtures/ pages from the manifest
uv run python -m medquery.ingest sources   # regenerate docs/SOURCES.md (a test checks it)
# Evaluation: not available yet (added in Phase 2)
```
Tesseract with English data is required (`tesseract --list-langs` shows `eng`). CI installs Ubuntu's `tesseract-ocr`.
Add a dependency with `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (dev), and add it in the phase that first needs it.

## Current phase
**Phase 0 (setup): done and approved.** CI is green on GitHub (origin: https://github.com/RajalingamMuthiah/MedQuery).
**Phase 1 (corpus and ingestion): done and approved.** CI is green (Tesseract OCR tests run on Ubuntu). It ingests 10 approved public-domain PDFs. Ruled tables (grid, open-sided and booktabs) become Markdown table chunks. Scanned pages are OCR'd. Chunks are section-aware, and short chunks merge within their section. Per-document stats, including mean OCR confidence, are in `data/processed/build_summary.json`.
Short chunks (under 30 words) are below the user's 8% threshold. The current rate is printed by `build` and stored in `build_summary.json`. Chunking will be judged by retrieval metrics in Phase 2.
Next: Phase 2 (baseline RAG and evaluation harness), once a plan is approved.

### Carry-forward rules for Phase 2 onwards
- **Retrieval excludes `kind = "quiz"` chunks by default.** These are Core Curriculum "Knowledge Check" items: questions with no answers, and some state false claims on purpose. Include them only behind an explicit opt-in, and never use them as evidence for an answer.
- Chunks carry `as_of` and `superseded`. Answers drawn from superseded guidance (JNC 7 Express) must say so.
- **Crash-safe outputs:** use `medquery/fileio.py` for every generated artifact, including eval reports. `atomic_write(path)` writes to a temp file and then calls `os.replace()`, so a crash never leaves a half-written file. `exclusive_lock(path)` refuses to start a second run. The build locks `data/processed/.build.lock`. If a crash leaves the lock behind, delete it once you have checked no run is active. Never run two builds at once: overlapping runs corrupted `chunks.jsonl` once during Phase 1.

### Ingestion design notes
- Text order comes from pdfminer text boxes, which handle two-column layouts. Running headers and footers, page numbers and table-of-contents pages are dropped.
- Headings come from font size (at least 1.15× the body size) or from short all-bold blocks. Bullets, URLs and phone numbers are never headings. A top-level heading shared by at least 80% of chunks is the document title and is removed from section paths.
- Tables: frames of touching ruling lines get closed with explicit outer edges. Booktabs tables are anchored on a "Table N" caption, with columns taken from where the rules are broken into segments. OCR pages get no table detection.
- Chunks: target 300 words and max 450 per packed group. Anything under 30 words merges into its smaller neighbour in the same section, which gives a hard ceiling of 480. Each table is one whole chunk.
- A short chunk alone in its section merges with an adjacent chunk under the same parent section. The merged chunk takes the parent path, and the subheading is kept as the first line of its text. Merges never cross top-level sections, documents or tables.
- Chunks in exercise sections ("Knowledge Check", "Self-Assessment", "Review Questions") get `kind = "quiz"`. This happens before junk filtering and merging, so they are never dropped and never merged with real content.
- Junk chunks are dropped, written to `data/processed/dropped.jsonl` with a reason for auditing, and counted by reason in the build summary: no real words, contents listings, and scattered labels (figures, axes, rotated text). A paragraph split by a column or page break is rejoined when prose stops mid-sentence and the next paragraph starts in lowercase.

## Deviations log
- Phase 0: `python-dotenv` is deferred to Phase 2, when the API key is first read.
- Phase 0: one smoke test was added, because pytest exits with code 5 on an empty suite.
- Phase 0: `docs/brief/medquery-v2-handoff.md` is git-ignored and untracked (it has personal details). It is kept locally only.
- Phase 1: a fourth subcommand, `fixtures`, was added so committed fixtures can be re-created from the manifest.
- Phase 1: a third fixture (MMWR p.8) and booktabs table detection were added, because the MMWR dosing tables have horizontal rules only.
- Phase 1: these were added at the user's request after reviewing chunk quality: short-chunk merging (within a section, then with siblings), junk dropping, document-title stripping and table-of-contents skipping.
- Phase 1: the download User-Agent includes the repo URL, because CDC's firewall rejects bare tool names.
- Phase 1 (local environment only): Tesseract's English model was missing, so `eng.traineddata` from tessdata_fast 4.1.0 was added to `%LOCALAPPDATA%\Tesseract-OCR\tessdata`.
