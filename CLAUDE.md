# CLAUDE.md: MedQuery v2

Health-document question answering (RAG) with grounded citations, abstention and claim-level hallucination checking. The full brief is in [docs/brief/medquery-v2-claude-code-prompt.md](docs/brief/medquery-v2-claude-code-prompt.md). `docs/brief/medquery-v2-handoff.md` (local only, not tracked) is background only; do not act on it.

## Ground rules
- Plan first. Propose each phase's plan and wait for approval before writing code.
- One phase at a time. At the end of each phase: run the tests, show the results, update this file, commit, push to origin (never force-push), summarize in 10 lines or fewer, then STOP.
- Windows and PowerShell. Give PowerShell commands, use `pathlib` for paths, and write no bash-only scripts.
- Never commit secrets. Keys go in `.env` (git-ignored); `.env.example` is committed. The key name is `GROQ_API_KEY`. Never print, log, cache or commit it.
- Corpus: public-domain, openly licensed or synthetic documents only, and never real patient data. Record every document's source and licence in [docs/SOURCES.md](docs/SOURCES.md).
- The app shows a "not medical advice" notice and never gives a diagnosis or dosing instructions beyond what the sources say.
- Never fabricate metrics. Every number in the README must come from a script that can be re-run.
- Small, readable modules. Use LangChain only where it removes code.
- Ask before deviating from the plan, and log approved deviations below.

## Stack
Python 3.11 (managed by uv), FastAPI, pytest, ruff, pre-commit. Retrieval: BM25 plus dense embeddings (BGE-class model) in FAISS or local Qdrant, reciprocal rank fusion, then a cross-encoder reranker. Embeddings: `fastembed` (ONNX) with `BAAI/bge-small-en-v1.5`; numpy cosine index. Generation and judging: **Groq** through its OpenAI-compatible endpoint (`openai` SDK, `base_url=https://api.groq.com/openai/v1`), behind the provider interface in `medquery/llm/`. Models: generator `openai/gpt-oss-20b`, judge `openai/gpt-oss-120b` (production models only), golden-set author `qwen/qwen3.8-27b` (preview; a one-time frozen generation only). Frontend: React, Vite and Tailwind. Packaging: Docker Compose and GitHub Actions.

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

# RAG and evaluation (needs GROQ_API_KEY in .env; every LLM call is cached in data/cache/llm)
uv run python -m medquery.ask "question"               # baseline answer with citations
uv run python -m medquery.eval golden generate --dry-run  # show sampled source pages, no LLM calls
uv run python -m medquery.eval golden generate          # candidates + review CSV (resumable)
uv run python -m medquery.eval golden topup --batch topup2  # top-ups append to review CSV (each runs once)
uv run python -m medquery.eval golden freeze --version v1 --reviewer "..." [--exclude ids] [--check]
uv run python -m medquery.eval run --split dev --config baseline   # resumable; writes medquery/eval/reports/
uv run python -m medquery.eval judge-sample --run <run_id>          # judge_review_v1.csv for hand labels
```
Tesseract with English data is required (`tesseract --list-langs` shows `eng`). CI installs Ubuntu's `tesseract-ocr`.
Add a dependency with `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (dev), and add it in the phase that first needs it.

## Current phase
**Phase 0 (setup): done and approved.** CI is green on GitHub (origin: https://github.com/RajalingamMuthiah/MedQuery).
**Phase 1 (corpus and ingestion): done and approved.** CI is green (Tesseract OCR tests run on Ubuntu). It ingests 10 approved public-domain PDFs. Ruled tables (grid, open-sided and booktabs) become Markdown table chunks. Scanned pages are OCR'd. Chunks are section-aware, and short chunks merge within their section. Per-document stats, including mean OCR confidence, are in `data/processed/build_summary.json`.
Short chunks (under 30 words) are below the user's 8% threshold. The current rate is printed by `build` and stored in `build_summary.json`. Chunking will be judged by retrieval metrics in Phase 2.
**Phase 2: Part B1 done, at STOP 2.** golden_v1 was frozen on 2026-10-22 (67 items: 15 unanswerable, 44 dev / 23 test, sha256 `ed7ec824…`). g097 and g098 (topup2, unreviewed) were excluded at freeze, at the user's choice. The dev baseline run is `eb2f931e39c1`; its provisional report `medquery/eval/reports/2026-10-22_baseline_dev.md` was made on a DIRTY tree and is flagged "judge NOT YET VALIDATED". `medquery/eval/golden/judge_review_v1.csv` (40 claims, 20 answers; verdicts hidden in the key file) awaits the user's labels.
Next, Part B2:
1. `judge-review freeze`, then agreement and kappa (judge_v2 only if claim kappa < 0.6).
2. Re-run dev from a clean commit for the final report.
3. One test-split run.
4. README metric definitions.
**Phase 2 (baseline RAG and evaluation harness): Part A done (STOP 1 passed).** The LLM layer, dense baseline RAG, `ask`, evidence mapping and golden generator are built. `medquery/eval/golden/review_v1.csv` has 96 candidates and awaits the user's hand review; see `medquery/eval/golden/README.md`. g001–g086 come from the first run. g087–g096 are top-up 1, added at the user's request: 5 tuberculosis unanswerable, 3 substantive superseded questions (prompt `golden_gen_v1+superseded_v2`), and 2 from the 1964 scan.
Next: Part B1 (freeze `golden_v1`, split, runner, judge, dev run, `judge_review_v1.csv`) after the review. Then STOP 2.

### Carry-forward rules for Phase 2 onwards
- **Retrieval excludes `kind = "quiz"` chunks by default.** These are Core Curriculum "Knowledge Check" items: questions with no answers, and some state false claims on purpose. Include them only behind an explicit opt-in, and never use them as evidence for an answer.
- Chunks carry `as_of` and `superseded`. Answers drawn from superseded guidance (JNC 7 Express) must say so.
- **Judge prompt (`judge_v1`):** `supported_by_context` and `supported_by_cited` are judged ONLY against the retrieved passages, never against the reference answer. The reference answer goes in a separate final section, used only for `correctness`, with an explicit instruction not to use it for support decisions. A test checks that it comes after the passages and the claims instructions.
- **Rate limits (Groq, per model):** 8K tokens/min and 1K requests/day (confirmed from headers); 30 requests/min and about 200K tokens/day (documented). All LLM calls go through the throttle, retries and disk cache in `medquery/llm/`. Eval runs are resumable.
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
- Phase 2: **the provider changed from Anthropic Claude to Groq**, at the user's request (the brief specifies Anthropic). The key is `GROQ_API_KEY`, accessed through the OpenAI-compatible endpoint behind `medquery/llm/`, so other providers can be added later.
- Phase 2: the unanswerability check shows the judge 6 dense + 6 keyword passages, not the planned 10 + 10, because 20 passages exceed the 8K tokens/min limit in a single request.
- Phase 2 (user-approved): a candidate question that copies 6 or more consecutive words from its page gets one rewrite by the author model. Then it is re-validated, or rejected. The original wording is kept as `original_question`.
- Phase 2 (user request): top-up 2 adds strict tuberculosis unanswerables. It uses author template `unanswerable_v2` (no false premises) and check `unanswerable_check_v2`:
  1. The judge lists the question's presuppositions, and each key term must occur in the non-quiz corpus.
  2. The answer search covers the top 15 dense plus top 15 keyword passages.
  3. A question is accepted only if no passage answers it and every presupposition is supported.
  The 30 passages go to the judge in batches of about 4.5K tokens, because one request must stay under 8K tokens/min.
- Phase 2 (**deviation from the brief**): the golden set was reviewed by an AI pre-review, not checked by hand by the user. The brief requires the user to check at least 25% by hand. `golden_v1.meta.json` records `"reviewer": "Claude (Anthropic) pre-review against source PDFs"`, as the user instructed. Reports and the README must say so; never describe golden_v1 as human-reviewed.
- Phase 2: topup2's false-premise key-term check proved too literal. The judge returns paraphrased terms ("daily dosage", "5% to 10%") that aren't verbatim in the corpus, so most of its 12 "false premise" rejections give the wrong reason. Only 2 of 6 were accepted, and both were excluded at freeze. Don't reuse `unanswerable_check_v2` without fixing the term matching.
- Phase 2 (user-approved): the judge-review sample adds **mismatched-context probes**. The dev run had only 39 claims, 1 of them unsupported, too few to validate hallucination detection.
  - **How they're made:** 8 dev answers were re-judged against passages retrieved for a different-topic dev question, giving 13 unsupported probe claims that fill the unsupported third of the 40-claim sample. Supported claims are all natural.
  - **How they're tracked:** probe rows are marked in `judge_review_v1.key.json`, and the agreement block reports `n_probe`.
- Phase 2: evidence normalisation ignores whitespace between a digit and the letter that follows ("130/80 mmHg" matches "130/80mmHg"), because PDF text layers drop such spaces. It's applied to both quotes and chunks.
- Phase 2: top-ups append to `review_v1.csv` byte for byte (`append_review_csv`), so the user's edited rows are never rewritten.
- Phase 2: the retry, throttle and cache logic lives in `medquery/llm/client.py`, separate from the protocol in `base.py`, so it can be tested with a fake transport.
- Phase 1 (local environment only): Tesseract's English model was missing, so `eng.traineddata` from tessdata_fast 4.1.0 was added to `%LOCALAPPDATA%\Tesseract-OCR\tessdata`.
