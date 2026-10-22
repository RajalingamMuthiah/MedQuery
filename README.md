# MedQuery v2

Question answering over public health documents, with cited answers, abstention when evidence is weak, and claim-level hallucination checking.

> **Not medical advice.** MedQuery answers only from its source documents and does not diagnose conditions or give dosing instructions. Always consult a qualified health professional.

## Setup (PowerShell)

Requires [uv](https://docs.astral.sh/uv/) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) with English language data (`tesseract --list-langs` must show `eng`).

```powershell
uv sync
Copy-Item .env.example .env   # then add your ANTHROPIC_API_KEY
uv run pytest
```

## Corpus and ingestion

The corpus is 10 public-domain US government documents on diabetes, hypertension and tuberculosis, listed with sources and licences in [docs/SOURCES.md](docs/SOURCES.md). The single source of truth is [corpus/manifest.toml](corpus/manifest.toml), which pins every file by sha256.

```powershell
uv run python -m medquery.ingest download  # fetch PDFs into data/raw/ and verify sha256
uv run python -m medquery.ingest build     # data/processed/chunks.jsonl + build_summary.json
```

The build keeps page numbers, section headings and tables, falls back to OCR on scanned pages, and prints a per-document summary that includes mean OCR confidence.

## Baseline RAG and evaluation

The baseline is deliberately simple:
- **Retrieval:** dense only, using `BAAI/bge-small-en-v1.5` via fastembed, top 5. Quiz chunks are excluded.
- **Generation:** one prompt (`answer_v1`) to `openai/gpt-oss-20b` on Groq. It answers only from the passages, cites chunk ids, and abstains when the passages don't contain the answer.
- **Judging:** OpenAI `gpt-6.1-sol`. It needs `OPENAI_API_KEY` in `.env`, while generation needs `GROQ_API_KEY`.

```powershell
uv run python -m medquery.ask "What is the rifampin dose for adults with latent TB?"
uv run python -m medquery.eval run --split dev --config baseline    # resumable, cached
uv run python -m medquery.eval run --split test --config baseline   # milestones only; clean tree
```

Reports go to [medquery/eval/reports/](medquery/eval/reports/). Each records:
- the run date, split, git commit and dirty flag;
- every model id and prompt hash;
- the corpus and golden sha256;
- the judge's agreement with hand labels.

### Evaluation data
- **Golden set `golden_v1`:** 67 questions (15 unanswerable), split 44 dev / 23 test, stratified by topic and type.
  - Questions were written by `qwen/qwen3.8-27b` from source pages, and every gold answer is anchored as (document, page, exact quote).
  - Tuning uses the dev split only; the test split runs only at milestones.
- **Judge validation:** 60 units (40 claims, 20 answers), hand-labelled by Rajalingam Muthiah.
  - 13 of the claims are *mismatched-context probes*: an answer judged against another question's passages. They're used because natural answers rarely contain unsupported claims.
  - Reports give % agreement and Cohen's kappa, separately for natural claims, probes and answers.

### Baseline results (Phase 2)
These are copied from the two reports below, which were produced from clean commits. Re-running the same command reproduces them exactly, because every LLM call is cached. Values are shown with their 95% bootstrap CI.

| Metric | Dev (44 questions) | Test (23 questions) |
|---|---|---|
| Answer correctness | 0.735 [0.588, 0.864] | 0.611 [0.368, 0.812] |
| Faithfulness | 0.875 [0.754, 0.968] | 0.909 [0.700, 1.000] |
| Claim hallucination rate | 0.095 [0.033, 0.177] | 0.033 [0.000, 0.137] |
| Context precision | 0.403 [0.289, 0.517] | 0.335 [0.173, 0.516] |
| Context recall | 0.676 [0.515, 0.817] | 0.611 [0.406, 0.824] |
| Recall@5 / recall@10 | 0.662 / 0.779 | 0.611 / 0.750 |
| MRR | 0.481 [0.348, 0.608] | 0.373 [0.226, 0.547] |
| Citation accuracy | 0.871 [0.787, 0.943] | 0.967 [0.863, 1.000] |
| Abstention accuracy | 0.841 [0.727, 0.932] | 0.696 [0.521, 0.870] |
| False-abstention rate | 0.206 [0.086, 0.364] | 0.389 [0.188, 0.632] |
| Latency p50 / p95, end to end | 0.69 s / 1.11 s | 0.75 s / 1.62 s |
| Generation cost per query (list price) | $0.000196 | $0.000184 |

Sources:
- dev: [2026-10-22_baseline_dev_e1f69bdb8e0a.md](medquery/eval/reports/2026-10-22_baseline_dev_e1f69bdb8e0a.md);
- test: [2026-10-22_baseline_test_b2bcda7e0fda.md](medquery/eval/reports/2026-10-22_baseline_test_b2bcda7e0fda.md).

The baseline's main weakness is **false abstention**: it declines answerable questions that its top-5 dense retrieval doesn't surface (recall@5 is about 0.6–0.7). Phase 3's retrieval upgrades target this.

### Metric definitions
Every quality metric has a 95% percentile bootstrap CI: 1,000 resamples over questions, seed 0. "Judge" means OpenAI `gpt-6.1-sol` with prompt `judge_v1`, used as an LLM judge (one call per answered question). Claim support is decided only from the retrieved passages, and the reference answer is used only for correctness.

| Metric | Definition | Computed over |
|---|---|---|
| Answer correctness | Judge compares the answer with the reference: correct 1, partial 0.5, incorrect 0. Abstaining on an answerable question scores 0 | answerable questions |
| Faithfulness | Per answer, the share of atomic claims supported by the retrieved passages, averaged over answers | answered questions |
| Claim hallucination rate | Unsupported claims ÷ all claims, pooled | answered questions |
| Answer relevancy | Judge rating 1–5, scaled to 0–1 | answered questions |
| Context precision | Rank-aware average precision over the 5 passages given to the generator. A passage is relevant if a gold quote maps to it | answerable questions |
| Context recall | Share of gold evidence quotes covered by the 5 passages | answerable questions |
| Recall@k (k = 1, 3, 5, 10) | Gold chunks in the top k ÷ gold chunks | answerable questions |
| MRR | 1 ÷ rank of the first gold chunk in the top 10 (0 if none) | answerable questions |
| Citation accuracy | Cited claims whose cited passages support them ÷ cited claims, pooled | answered questions |
| Citation validity | Citations that name a passage actually given to the generator ÷ all citations | answered questions |
| Abstention accuracy | Abstains on unanswerable questions and answers answerable ones. Abstention recall and the false-abstention rate are reported too | all questions |
| Latency p50/p95 | Retrieval (query embedding plus search), generation (the original API call), and end-to-end | all questions |
| Cost per query | Generator tokens × Groq list price (dated in `medquery/config.py`). Judge cost is reported separately as evaluation overhead | all questions |

Gold evidence is stored as (document, page, quote), never as chunk ids. Each quote is mapped to chunks at evaluation time, so the metrics survive re-chunking, and mapping coverage is reported in every report.

## Known limitations

- **Superseded guidance.** JNC 7 Express (2003) is superseded by later hypertension guidance, notably the 2017 ACC/AHA guideline, which lowered the hypertension threshold from 140/90 to 130/80 mmHg. Its chunks carry an `as_of` date and a `superseded` note.
- **Tables are detected only when ruled.** Grid tables, open-sided tables and caption-anchored "booktabs" tables (horizontal rules only) become table chunks. Tables laid out with whitespace alone stay as body text.
- **Scanned pages are OCR'd as plain text.** No table structure is recovered from scans. The 1964 tuberculosis report is mostly tables, so its OCR quality is lower. Confidence is reported in the build summary rather than hidden.
- **Heading detection is heuristic,** based on font size and bold weight. OCR pages have no headings. Mis-levelled headings occur: for example, NDSR 2020's large "SUGGESTED CITATION" heading becomes the parent of its appendices.
- **Floating tables take the section where they appear in reading order.** That can be the subsection just before them rather than the one that discusses them (MMWR Table 4 is one example).
- **Quiz content is kept but labelled.** The Core Curriculum's "Knowledge Check" items are unanswered questions, and some state false claims on purpose. They are stored as `kind = "quiz"` and excluded from retrieval by default.
- **OCR errors remain in scanned text.** Examples are dropped decimal points and misread years. Each chunk carries its OCR confidence.
- **The golden set was reviewed by an AI, not by a person.** golden_v1 (67 questions) was written by `qwen/qwen3.8-27b`, then pre-reviewed by Claude (Anthropic) against the source PDFs. No human hand-check is recorded. Some strata are thin: only 2 tuberculosis unanswerable questions and 3 superseded ones. Treat per-stratum results as indicative.
- **The LLM judge is only partly validated.** Natural answers rarely contain unsupported claims: the hand-labelled sample has 3. So Cohen's kappa on natural claims is unstable, with confidence intervals spanning most of 0 to 1, and the raw catch count (how many human-flagged unsupported claims the judge also flagged) is reported alongside it. Probe claims are easy cases and overstate agreement on their own.
- **The judge runs at temperature 1.** The judge is OpenAI `gpt-6.1-sol`, which only accepts its default temperature. Judge outputs are cached, so re-running a report reproduces it exactly. A stability check (the same labelled set judged with a second seed) changed 0 of 39 claim verdicts and 0 of 20 answer verdicts.
- **judge_v2's exact-match rule was written after seeing these hand labels,** so its measurement on them is optimistic. On gpt-6.1-sol it gave no measured gain over judge_v1, which is the prompt in use. judge_v2 will be re-validated on a fresh labelled sample in Phase 3.
- **The judge is strict on interpretive claims.** On natural claims, the judge rejected 2 that the labeller marked supported. Each was a part of the answer that went beyond the passage wording (for example "the primary driver"). Faithfulness and hallucination rate may therefore be slightly pessimistic.
- **Small samples.** 44 dev and 23 test questions give wide confidence intervals. Differences between configurations smaller than the CIs shouldn't be read as real.
- **Small, US-focused corpus.** It has three topics and ten documents, mostly from US federal agencies.
