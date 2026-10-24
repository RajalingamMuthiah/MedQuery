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
Python 3.11 (managed by uv), FastAPI, pytest, ruff, pre-commit. Retrieval: BM25 plus dense embeddings (BGE-class model) in FAISS or local Qdrant, reciprocal rank fusion, then a cross-encoder reranker. Embeddings: `fastembed` (ONNX) with `BAAI/bge-small-en-v1.5`; numpy cosine index. LLMs go through the provider router in `medquery/llm/` (`openai` SDK; each model is mapped to its provider in `medquery/config.py`).
- **Generator:** `openai/gpt-oss-20b` on **Groq** (`GROQ_API_KEY`) through Phase 3; on **NVIDIA NIM** (`NVIDIA_API_KEY`, same model id) from Phase 4, chosen per run by `RunConfig.generator_provider`.
- **Judge:** `gpt-6.1-sol` on **OpenAI** (`OPENAI_API_KEY`), prompt `judge_v1`. The model accepts only its default temperature, so `MODEL_OVERRIDES` omits `temperature` for it.
- **Golden-set author:** `qwen/qwen3.8-27b` on Groq (preview; a one-time frozen generation only).
- **Earlier judge:** `openai/gpt-oss-120b` on Groq, kept in the report history. Frontend: React, Vite and Tailwind. Packaging: Docker Compose and GitHub Actions.

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
uv run python -m medquery.eval best-row --configs <names>          # best-row rule over completed dev runs
uv run python -m medquery.eval ablation --split dev --configs <names>  # paired ablation report (clean tree, dev only)
uv run python -m medquery.eval judge-sample --run <id> --version v2 --exclude-labelled v1  # fresh sample
uv run python -m medquery.eval judge-review freeze --version v2 --labeller "..."
uv run python -m medquery.eval judge-review measure --version v2 --judge judge_v2
```
Tesseract with English data is required (`tesseract --list-langs` shows `eng`). CI installs Ubuntu's `tesseract-ocr`.
Add a dependency with `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (dev), and add it in the phase that first needs it.

## Current phase
**Phase 0 (setup): done and approved.** CI is green on GitHub (origin: https://github.com/RajalingamMuthiah/MedQuery).
**Phase 1 (corpus and ingestion): done and approved.** CI is green (Tesseract OCR tests run on Ubuntu). It ingests 10 approved public-domain PDFs. Ruled tables (grid, open-sided and booktabs) become Markdown table chunks. Scanned pages are OCR'd. Chunks are section-aware, and short chunks merge within their section. Per-document stats, including mean OCR confidence, are in `data/processed/build_summary.json`.
Short chunks (under 30 words) are below the user's 8% threshold. The current rate is printed by `build` and stored in `build_summary.json`. Chunking will be judged by retrieval metrics in Phase 2.
**Phase 2 (baseline RAG and evaluation harness): done and approved (2026-10-22).**
- **Final reports, from clean commits:**
  - dev `medquery/eval/reports/2026-10-22_baseline_dev_e1f69bdb8e0a.md`: answer correctness 0.735, recall@5 0.662, abstention accuracy 0.841;
  - test `…_test_b2bcda7e0fda.md`: correctness 0.611, false abstention 0.389.
- **Main baseline weakness:** false abstention on answerable questions that retrieval misses.
- **Next:** Phase 3, retrieval upgrades (brief: `docs/brief/phase3-prompt.md`). BM25 with RRF, a reranker, then contextual chunk headers, as a cumulative ablation on **dev only**: the test split is not run in Phase 3. Each step is compared with paired bootstrap differences, and the phase ends with a fresh judge re-validation sample.

**Phase 3 (retrieval upgrades): done and approved (2026-10-24).**
- **Rows (dev only):** `baseline`, `hybrid`, `hybrid_rerank`, `hybrid_rerank_headers`, `hybrid_rerank_headers_maxp` (post-hoc; headers won), `hybrid_rerank_headers_maxp_k8`.
- **Best-row rule** (`best-row --configs ...`, recorded in the ablation report): highest dev answer correctness; ties broken by the lower false-abstention rate. Every k = 5 row is a candidate, including MaxP.
- **Comparisons:** paired bootstrap against the previous row; a k = 8 row against its k = 5 config; MaxP against the same config with truncated reranker input. Guards: `abstention_recall` and `citation_accuracy`.
- **Done (2026-10-24):** all six dev runs (44/44 each, reports committed); ablation report `medquery/eval/reports/2026-10-24_ablation_dev_9d8744145a.md`. The best row by the rule is `hybrid_rerank_headers_maxp` (run `a14cd47892bd`): answer correctness 0.941, false abstention 0.029. Real paired gains: `hybrid` vs `baseline` (correctness +0.176, false abstention -0.147) and headers vs `hybrid_rerank` (recall@5 +0.118, MRR +0.135). Reranker retrieval costs about 10-20 s per query on CPU.
- **Fresh judge sample `judge_review_v2`:** 25 natural claims and 10 answers from run `a14cd47892bd`, none in v1, hand-labelled by Rajalingam Muthiah without the key (labels sha256 `1c5d07a1…`).
  - **Labels r2** (`768e9c34…`, supersedes v1 `1c5d07a1…`): only r031 changed, to unsupported (the passage value 0.7 is in millions, not %). Meta records `revision_reason`: the re-check was prompted by a judge disagreement, so r2 may be biased toward the judge. Both sets are kept and reported.
  - On r2, gpt-6.1-sol/judge_v1: claims 0.880 (kappa 0.658), answers 0.900 (kappa 0.630); natural unsupported caught 4 of 4; supported wrongly rejected 3 of 21 (r008, r009, r030). On v1 labels: 0.840 (kappa 0.519), 3 of 3, 4 of 22.
  - On r2, gpt-6.1-sol/judge_v2: claims 0.720 (kappa 0.390); wrongly rejected 7 of 21 (v1 labels: 0.680, 8 of 22). judge_v1 stays.
  - The judge is stricter than the labeller, so faithfulness and hallucination rate are pessimistic (README limitations).
- **Next:** Phase 4, generation quality and the live config (brief: `docs/brief/phase4-prompt.md`). The plan is to be proposed. Candidates from the dev diagnosis: table chunks losing units (r031: 0.7 million read as 0.7%), citation accuracy drifted down (0.871 to 0.740, no single real step), and the judge's false rejections. Choose the live retrieval config with latency in mind: `hybrid` is within every later row's CI at ~14 ms, while the reranker costs 10-20 s per query on CPU.
**Phase 4 (generation quality and the live config): in progress** (plan approved 2026-10-24; brief `docs/brief/phase4-prompt.md`).
- **Provider switch:** generator on NVIDIA NIM (`https://integrate.api.nvidia.com/v1`, model `openai/gpt-oss-20b`). Probe 2026-10-24: the Groq settings return 200, but so does an unknown parameter (NIM ignores what it does not support); `reasoning_effort` is honoured; three temperature-0/seed-0 repeats were identical. No rate-limit headers (client limit 35 RPM, conservative), no reasoning-token breakdown in usage, some calls queue for over 60 s. NVIDIA generation is costed at $0 (free API-catalog credits, no list price).
- **Judge budget:** Phase 4 runs (generator on NVIDIA) stop before starting if the phase's OpenAI judge spend plus the run's estimate (x1.5) would exceed $3 (`medquery/eval/budget.py`); every Phase 4 report prints the running total.
- **Plan:** provider check (`hybrid`, `hybrid_rerank_headers_maxp` on NVIDIA; stop if any real difference), then cumulative rows `hybrid_headers` -> corpus v2 table fix -> `answer_v2` -> abstention by retrieval confidence (offline), dev only.
- **Provider check done** (`medquery/eval/reports/2026-10-24_provider_check_dev_6e6b357b58.md`): 24 paired tests (2 rows x 12 metrics). One was "real": faithfulness on `hybrid_rerank_headers_maxp` +0.016 on NVIDIA, CI [0.0003, 0.036]. That is expected by chance across 24 tests, and it favours NVIDIA. Answer correctness, false abstention, abstention recall and every retrieval metric are unchanged. The user decided (2026-10-24) to treat the switch as neutral and continue the Phase 4 rows on NVIDIA.
- **Corpus v2** (`data/processed/v2/`, `ingest build --corpus v2`) passed its gate: prose chunks identical to v1, all 71 gold quotes map exactly on the same pages.
**Phase 2 Part B2 (2026-10-22):** judge switched to OpenAI `gpt-6.1-sol` with `judge_v1`, at the user's decision.
- **Hand labels:** `labels` (v1, `ca799c8b…`), then `labels_r2` (`8d2ee727…`). r2 corrects r022, r027, r028 and r050 after the user re-checked them.
- **Measurements on r2** (raw counts use matched units only):
  - gpt-oss-120b/judge_v1 caught 1 of 3 unsupported natural claims and wrongly rejected 0.
  - gpt-6.1-sol/judge_v1 and gpt-6.1-sol/judge_v2 each caught 3 of 3 and wrongly rejected 2 of 23 (r036, r037: strict on interpretive parts).
  - Answer agreement was 0.850 for all three.
- **Seed stability** (gpt-6.1-sol/judge_v1): 0 of 39 claim and 0 of 20 answer verdicts changed.
- **Why judge_v1:** v2 gave no measured gain, and the user's rule (at most 1 false rejection) selected neither, so v1 avoids the overfitting risk.
**Phase 2: Part B1 done (STOP 2 passed).** golden_v1 was frozen on 2026-10-22 (67 items: 15 unanswerable, 44 dev / 23 test, sha256 `ed7ec824…`). g097 and g098 (topup2, unreviewed) were excluded at freeze, at the user's choice. The dev baseline run is `eb2f931e39c1`; its provisional report `medquery/eval/reports/2026-10-22_baseline_dev.md` was made on a DIRTY tree and is flagged "judge NOT YET VALIDATED". `medquery/eval/golden/judge_review_v1.csv` (40 claims, 20 answers; verdicts hidden in the key file) awaits the user's labels.
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
- **Rate limits:**
  - **Groq, per model:** 8K tokens/min and 1K requests/day (from headers); 30 requests/min and 200K tokens/day (documented, and hit on 2026-10-22).
  - **OpenAI:** 10K requests/min and 4M tokens/min (from headers); we throttle to 500 requests/min and 400K tokens/min.
  - All LLM calls go through the router, throttle, retries and disk cache in `medquery/llm/`. Eval runs are resumable.
- **Judge measurements on frozen labels:** a different judge model splits claims differently. Each labelled claim maps to the re-extracted claims that make it up (`matching_parts`: content-word containment of at least 0.7, otherwise text similarity of at least 0.6). It is unsupported if any part is. Unmatched units are reported, never guessed.
- **Crash-safe outputs:** use `medquery/fileio.py` for every generated artifact, including eval reports. `atomic_write(path)` writes to a temp file and then calls `os.replace()`, so a crash never leaves a half-written file. `exclusive_lock(path)` refuses to start a second run. The build locks `data/processed/.build.lock`. If a crash leaves the lock behind, delete it once you have checked no run is active. Never run two builds at once: overlapping runs corrupted `chunks.jsonl` once during Phase 1.

### Carry-forward for Phase 6 (live API)
- **Serve generation on Groq; evaluate on NVIDIA.** NVIDIA free-tier generation latency is 12-16 s median and 58-72 s p95, against about 0.8 s on Groq (provider check, 2026-10-24). The provider check found no change in answer correctness, false abstention, abstention recall or retrieval, which justifies running evaluation on NVIDIA while the live app generates on Groq.
- **MaxP reranker window:** `MAXP_WINDOW = 320` tokens (overlap 96) in `medquery/eval/runner.py` assumes questions of at most 56 tokens (the longest golden question) and headers of at most 115 tokens (the longest in the corpus): 320 + 115 + 56 + 3 special tokens = 494, under bge-reranker-base's 512. Live user questions can be longer. Compute the window from the actual question (and header) length per request, or cap the question length, so every reranker input stays under 512 tokens.

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
- Phase 2 (**deviation, user decision 2026-10-22**): the judge moved permanently from Groq `openai/gpt-oss-120b` to OpenAI `gpt-6.1-sol`. The generator stays on Groq.
  - **Reasons:** Groq's 200K tokens/day quota blocked Part B2, and a different family from the generator removes self-preference bias.
  - **How it was chosen:** from the live `/v1/models` list and the OpenAI pricing and deprecation pages. It is current, mid-cost ($2/$10 per 1M), and supports JSON mode and seed.
  - **Temperature:** no current OpenAI model accepts temperature 0. gpt-4.1 does, but shuts down on 2027-01-14. A seed stability check stands in for determinism.
  - The interrupted judge_v2 measurement on gpt-oss-120b (Groq quota) was stopped. Its cached calls are kept, and it was not completed.
- Phase 2: the measurement method for a new judge on frozen labels changed after the first gpt-6.1-sol run. Whole-sentence matching left 8 of 60 units unmatched because of claim granularity; part-based matching left 1. All measurements were re-run from the cache, with no new API calls.
- Phase 2: evidence normalisation ignores whitespace between a digit and the letter that follows ("130/80 mmHg" matches "130/80mmHg"), because PDF text layers drop such spaces. It's applied to both quotes and chunks.
- Phase 2: top-ups append to `review_v1.csv` byte for byte (`append_review_csv`), so the user's edited rows are never rewritten.
- Phase 2: the retry, throttle and cache logic lives in `medquery/llm/client.py`, separate from the protocol in `base.py`, so it can be tested with a fake transport.
- Phase 3 (user request): the reranker truncation share (inputs over 512 tokens) is reported per row, with a note above 10%, and multi-chunk questions get their own breakout.
- Phase 3 (**post-hoc row, user request**): `hybrid_rerank_maxp` (or `hybrid_rerank_headers_maxp`, the same with headers; both are defined and one runs) was added after dev showed 34.9% of `hybrid_rerank` inputs were truncated. It scores each passage as the max over 320-token windows (overlap 96), with the header on each window only if `hybrid_rerank_headers` beat `hybrid_rerank` by the best-row rule. The ablation report labels it post-hoc.
- Phase 3 (user request): the ablation report includes `abstention_recall` and `citation_accuracy` with paired differences, as guards.
- Phase 3 (user request): the remaining dev runs run unattended in a detached PowerShell process, `data/scripts/phase3_runs.ps1` (git-ignored), logging to `data/logs/phase3_runs.log`. It checks the git tree is clean before each run and aborts otherwise, and commits each finished run's report locally (no push).
- Phase 4 (lesson, not a retroactive change): a stop rule that fires on any of 24 paired tests raises chance false alarms (about 1 expected at 95%). Future provider or config checks should stop only on the primary metrics: answer correctness, false-abstention rate and hallucination rate. The Phase 4 provider check and its report are left as run.
- Phase 3 (user request): `judge-review freeze --reason` records why a label revision's rows were re-checked (`revision_reason` in the meta).
- Phase 4 (brief): the generator moves from Groq to NVIDIA NIM (same model). Earlier runs keep their ids: `generator_provider` at its default (`groq`) is left out of the run id.
- Phase 4 (user decisions): full table_v2 fix in one row; abstention operating point = max overall dev score with a leave-one-out estimate; citation parser normalises Unicode hyphens and fullwidth brackets, and Phase 4 reports re-parse citations for every row.
- Phase 3: code changes made while a dev run was in progress went into a separate git worktree (branch `phase3-maxp`), then were fast-forwarded into main, so every report records the clean commit it ran from.
- Phase 1 (local environment only): Tesseract's English model was missing, so `eng.traineddata` from tessdata_fast 4.1.0 was added to `%LOCALAPPDATA%\Tesseract-OCR\tessdata`.
