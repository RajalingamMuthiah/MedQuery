# MedQuery v2

Question answering over public health documents, with cited answers, abstention when evidence is weak, and claim-level hallucination checking.

> **Not medical advice.** MedQuery answers only from its source documents and does not diagnose conditions or give dosing instructions. Always consult a qualified health professional.

## Final results (test split)

These are the headline numbers. Each config ran on the 23-question test split **exactly once**, after every design decision had been made on the dev split, and nothing was tuned after seeing them. Every number is copied from [2026-10-25_final_test_7e48e89de6.md](medquery/eval/reports/2026-10-25_final_test_7e48e89de6.md), produced from a clean commit:

```powershell
uv run python -m medquery.eval ablation --split test --final --configs baseline_nvidia hybrid_nvidia hybrid_headers_tables_answer_v2 live_verify
```

All rows use the same generator (`openai/gpt-oss-20b`, run on NVIDIA NIM for evaluation) and the same judge (`gpt-6.1-sol` with `judge_v1`). Values are given with their 95% bootstrap CI.

| Config | Answer correctness | False-abstention rate | Hallucination rate | Citation accuracy | Recall@5 | End-to-end p50 / p95 |
|---|---|---|---|---|---|---|
| Baseline: dense top 5, `answer_v1` | 0.556 [0.333, 0.750] | 0.389 [0.188, 0.632] | 0.059 [0.000, 0.167] | 0.920 | 0.611 | 12.9 s / 71.6 s |
| Hybrid: dense + BM25 with RRF | 0.667 [0.444, 0.882] | 0.222 [0.053, 0.438] | 0.125 [0.000, 0.321] | 0.857 | 0.722 | 2.9 s / 33.3 s |
| **Live config**: hybrid + headers + table fix + `answer_v2` | **0.833 [0.656, 0.972]** | 0.111 [0.000, 0.286] | 0.158 [0.030, 0.346] | 0.842 | 0.750 | 1.9 s / 5.7 s |
| Live config + claim verifier | 0.833 [0.656, 0.972] | 0.111 [0.000, 0.286] | 0.158 [0.030, 0.346] | 0.842 | 0.750 | 6.8 s / 21.5 s |

Paired differences (each row against the one above it, on the same questions; 1,000 resamples, seed 0):
- **Live config vs hybrid: answer correctness +0.167, 95% CI [0.029, 0.324]**. This is the one real difference on test.
- Hybrid vs baseline: answer correctness +0.111 [0.000, 0.237] and false-abstention rate −0.167 [−0.353, 0.000]. Both fall just short of real.
- **Verifier: no change on test.** It removed no sentences, 1 of its 16 verifications failed (that answer was kept unverified), and the judge still found 6 unsupported claims in the verified answers.
- **Abstention recall is 1.000 in every row**: the system declined every unanswerable test question.

**Summary comparison computed after the final runs; no tuning.** This compares the live config directly with the baseline, from the stored records of the same two test runs ([2026-10-25_summary_test_dacaeb434340_vs_ff63d760267f.md](medquery/eval/reports/2026-10-25_summary_test_dacaeb434340_vs_ff63d760267f.md), `summary-compare --split test --configs baseline_nvidia hybrid_headers_tables_answer_v2`):

| Metric | Baseline | Live config | Difference [95% CI] | Real? |
|---|---|---|---|---|
| Answer correctness | 0.556 | 0.833 | +0.278 [0.118, 0.469] | **yes** |
| False-abstention rate | 0.389 | 0.111 | −0.278 [−0.479, −0.100] | **yes** |
| Hallucination rate | 0.059 | 0.158 | +0.099 [−0.030, 0.265] | no |

**In the live app the claim verifier is off by default.** It changed nothing on test and adds latency. It stays available as an optional, asynchronous "check this answer" feature.

**Read these with care:**
- **The test split is small:** 23 questions, 18 of them answerable, so the CIs span 0.3 or more.
- **Hallucination rate rises down the table** (0.059 → 0.158), but no step's change is real. The judge is also stricter than the human labeller: on a fresh hand-labelled sample it wrongly rejected 3 of 21 supported claims (see the judge re-validation below).
- **The golden set was written by an LLM and reviewed by an AI,** not by a person.
- **Generation runs on NVIDIA's free tier for evaluation**, which is slow and queues. The live app will generate on Groq (about 0.8 s per answer).

The development history below (Phases 2–5, dev split) records how each choice was made. Dev results are not test results.

## Setup (PowerShell)

Requires [uv](https://docs.astral.sh/uv/) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) with English language data (`tesseract --list-langs` must show `eng`).

```powershell
uv sync
Copy-Item .env.example .env   # then add GROQ_API_KEY and NVIDIA_API_KEY (generator) and OPENAI_API_KEY (judge)
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

### Retrieval upgrades (Phase 3, dev only)
Cumulative rows on the 44 dev questions (34 answerable). The generator, prompt `answer_v1`, judge (`gpt-6.1-sol`, `judge_v1`) and golden set are the same in every row. The test split was not run. Every number below is copied from [2026-10-24_ablation_dev_9d8744145a.md](medquery/eval/reports/2026-10-24_ablation_dev_9d8744145a.md), produced from a clean commit by:

```powershell
uv run python -m medquery.eval ablation --split dev --configs baseline hybrid hybrid_rerank hybrid_rerank_headers hybrid_rerank_headers_maxp hybrid_rerank_headers_maxp_k8
```

| Row | Retrieval | Answer correctness | False-abstention rate | Recall@5 | MRR | Retrieval latency p50 / p95 |
|---|---|---|---|---|---|---|
| `baseline` | dense top 5 | 0.735 [0.588, 0.864] | 0.206 [0.086, 0.364] | 0.662 | 0.481 | 8 ms / 12 ms |
| `hybrid` | + BM25, RRF (k = 60) | 0.912 [0.828, 0.985] | 0.059 [0.000, 0.143] | 0.779 | 0.533 | 14 ms / 29 ms |
| `hybrid_rerank` | + bge-reranker-base over the top 20 | 0.882 [0.763, 0.972] | 0.118 [0.028, 0.237] | 0.765 | 0.549 | 9.9 s / 12.4 s |
| `hybrid_rerank_headers` | + title and section headers on index text | 0.941 [0.857, 1.000] | 0.059 [0.000, 0.143] | 0.882 | 0.684 | 11.0 s / 15.7 s |
| `hybrid_rerank_headers_maxp` (post-hoc) | reranker scores 320-token windows (overlap 96), max per passage | 0.941 [0.857, 1.000] | 0.029 [0.000, 0.088] | 0.897 | 0.660 | 20.4 s / 27.3 s |
| `hybrid_rerank_headers_maxp_k8` | the best row, 8 passages to the generator | 0.926 [0.838, 1.000] | 0.029 [0.000, 0.088] | 0.897 | 0.660 | 16.7 s / 21.5 s |

How to read it:
- **Paired bootstrap.** Each row is compared with the previous row on the same questions (a k = 8 row with its k = 5 config, the MaxP row with the same config truncating its input); 1,000 resamples, seed 0. A difference counts as real only if its 95% CI excludes 0.
- **Real gains:** `hybrid` over `baseline` (answer correctness +0.176 [0.062, 0.309]; false-abstention rate -0.147 [-0.281, -0.054]), and headers over `hybrid_rerank` (recall@5 +0.118 [0.014, 0.244]; MRR +0.135 [0.040, 0.239]). No other step changed a metric measurably.
- **The reranker adds 10–20 s per query on CPU with no measurable accuracy gain.** `hybrid_rerank` vs `hybrid` shows no real difference in any metric. Without windowing, 34.9% of its inputs (41.9% with headers) exceed bge-reranker-base's 512-token limit and are truncated.
- **MaxP was added after seeing the dev truncation**, so it is a post-hoc row. It removes truncation (0 of 1,713 inputs) but changes no metric measurably, and roughly doubles reranker latency.
- **k = 8** changed no metric measurably; on the 5 multi-chunk questions, 4 already had both gold passages in context at k = 5.
- **Guards:** abstention recall (unanswerable questions declined) is 1.000 in every row. Citation accuracy drifts from 0.871 (baseline) to 0.740 (k = 8), but no single step's drop is real.
- **Best row by the stated rule** (highest answer correctness, ties broken by the lower false-abstention rate): `hybrid_rerank_headers_maxp`. But `hybrid`'s answer correctness (0.912) lies inside the 95% CI of every later row, at under 0.2% of their retrieval latency.

### Judge re-validation on a fresh sample (Phase 3)
A second sample, `judge_review_v2`, was drawn from the best dev run: 25 claims and 10 answers, none of them in the first sample, hand-labelled by Rajalingam Muthiah without seeing the judge's verdicts. All claims are natural (no probes were needed).

| Judge | Labels | Claim agreement [95% CI] | Claim kappa [95% CI] | Natural unsupported caught | Natural supported wrongly rejected | Answer agreement [95% CI] | Answer kappa |
|---|---|---|---|---|---|---|---|
| `gpt-6.1-sol` + `judge_v1` (in use) | r2 (current) | 0.880 [0.760, 1.000] | 0.658 [0.233, 1.000] | 4 of 4 | 3 of 21 | 0.900 [0.700, 1.000] | 0.630 |
| `gpt-6.1-sol` + `judge_v2` | r2 (current) | 0.720 [0.520, 0.880] | 0.390 [0.109, 0.719] | 4 of 4 | 7 of 21 | 0.900 [0.700, 1.000] | 0.630 |
| `gpt-6.1-sol` + `judge_v1` | v1 (first labels) | 0.840 [0.680, 0.960] | 0.519 [0.000, 0.884] | 3 of 3 | 4 of 22 | 0.900 [0.700, 1.000] | 0.630 |
| `gpt-6.1-sol` + `judge_v2` | v1 (first labels) | 0.680 [0.480, 0.840] | 0.296 [0.000, 0.606] | 3 of 3 | 8 of 22 | 0.900 [0.700, 1.000] | 0.630 |

- **Label revision r2 changes one row, r031,** from supported to unsupported: the passage's "0.7 (0.4–1.0)" is in millions, not a percentage. The re-check was prompted by a judge disagreement, and only disputed rows were re-checked, so r2 may be biased toward the judge. Both label sets are kept and both are reported.
- From `medquery/eval/golden/judge_review_v2.measure_<judge>_gpt-6.1-sol_<r2|v1>.json` (`uv run python -m medquery.eval judge-review measure --version v2 --judge <judge_v1|judge_v2>` measures the latest labels). judge_v2's fresh validation is worse than judge_v1 on either label set, so judge_v1 stays.

### Generation quality and the live config (Phase 4, dev only)
The generator moved to NVIDIA NIM (same model, `openai/gpt-oss-20b`) for evaluation. Every number below is copied from four reports produced from clean commits:
- [provider check](medquery/eval/reports/2026-10-24_provider_check_dev_6e6b357b58.md);
- [failure diagnosis](medquery/eval/reports/2026-10-24_diagnosis_dev_a14cd47892bd.md);
- [risk–coverage](medquery/eval/reports/2026-10-24_risk_coverage_dev_53088e55be95.md);
- [Phase 4 ablation](medquery/eval/reports/2026-10-24_ablation_dev_0565b682a4.md).

**Provider check (Groq vs NVIDIA, same retrieval).** 24 paired tests (2 rows × 12 metrics). One was "real": faithfulness on `hybrid_rerank_headers_maxp`, +0.016 on NVIDIA with 95% CI [0.0003, 0.036]. That is expected by chance across 24 paired tests (about 1.2 at 95%), and it favours NVIDIA. All primary metrics are unchanged on both rows: answer correctness, false abstention, abstention recall and retrieval. The switch is treated as neutral. NVIDIA's free tier is slow, though: generation p50 / p95 is 12.1 s / 58.1 s on `hybrid`, against 0.72 s / 1.12 s on Groq.

**Diagnosis of the best Phase 3 run** (`a14cd47892bd`): 1 incorrect answer and 13 unsupported claims across 6 answers, plus 1 false abstention.

| Cause (hand-assigned, committed CSV) | Findings |
|---|---|
| table/units misread (g018: 0.7 million read as 0.7%) | 2 |
| detail not in the passages (g092) | 6 |
| citation pointing to the wrong passage | 1 |
| number/qualifier near-miss | 1 |
| multi-passage synthesis | 1 |
| paraphrase beyond the passage | 1 |
| retrieval: gold not in the top 10 | 1 |

**Table fix (corpus v2).** Tables only; v1 is untouched.

| Table chunks | Corpus v1 | Corpus v2 |
|---|---|---|
| table chunks | 207 | 219 |
| continuation without caption or header | 50 | 0 |
| caption cut mid-sentence | 5 | 0 |
| header rows above numeric data (not merged) | 29 | 15 |
| all-empty columns | 83 | 0 |
| split decimals ("26 .8") | 8 | 0 |

The fix completes wrapped captions, gives continuation pages their caption and header, merges multi-row headers so each column carries its unit ("Undiagnosed diabetes Number in Millions (95% CI)"), drops empty columns and rejoins split decimals.

The gate passed: the 926 prose chunks are identical to v1, and all 71 gold evidence quotes map exactly to chunks on the same pages.

**Cumulative rows (generator on NVIDIA).**

| Row | Change | Answer correctness | False-abstention rate | Hallucination rate | Citation accuracy | End-to-end p50 / p95 |
|---|---|---|---|---|---|---|
| `hybrid_nvidia` | hybrid (Phase 3 row 2) | 0.912 [0.828, 0.985] | 0.059 [0.000, 0.143] | 0.059 [0.013, 0.125] | 0.873 | 12.2 s / 58.1 s |
| `hybrid_headers` | + title/section headers, no reranker | 0.956 [0.892, 1.000] | 0.029 [0.000, 0.088] | 0.113 [0.033, 0.207] | 0.853 | 12.5 s / 77.5 s |
| `hybrid_headers_tables` | + corpus v2 (table fix) | 0.956 [0.892, 1.000] | 0.029 [0.000, 0.088] | 0.114 [0.029, 0.211] | 0.865 | 10.2 s / 55.5 s |
| `hybrid_headers_tables_answer_v2` | + `answer_v2` prompt | 0.882 [0.781, 0.973] | 0.059 [0.000, 0.139] | 0.057 [0.014, 0.112] | 0.910 | 9.8 s / 64.9 s |
| `…_answer_v2_abstain` | + abstention by retrieval confidence (offline) | 0.882 [0.781, 0.973] | 0.059 [0.000, 0.139] | 0.057 [0.014, 0.112] | 0.910 | 9.8 s / 64.9 s |

- **No paired difference in Phase 4 is real** (each row against the previous one; 1,000 resamples, seed 0).
- **answer_v2:** answer correctness −0.074 [−0.162, 0.000] and hallucination rate −0.057 [−0.160, 0.025]. It trades three answers for fewer unsupported claims (per-item answers are in the runs' committed `_items.csv` reports):
  - g012: it missed a fact that was in its passages;
  - g078: it declined to compute "120 doses", which no passage states;
  - g081: it named the wrong risk factor.
- **The table fix** changes no metric. Its target error, g018, is already answered correctly once headers are on, because Table 1a then ranks first.
- **Abstention by retrieval confidence:** the operating-point rule (maximise the overall dev score; signal = top-1 dense cosine) chose **no threshold**. The generator already declines every unanswerable dev question (abstention recall 1.000), so any threshold only removes correct answers. The leave-one-out estimate agrees (overall 0.909, no threshold chosen in any of the 44 folds).
- **End-to-end latency** here is dominated by NVIDIA's free-tier queue. Retrieval without the reranker takes 13–21 ms at the median.

**Live config (decided after Phase 4):** hybrid retrieval + title/section headers + corpus v2 + `answer_v2`, with no reranker. Generation is served on Groq for latency; evaluation runs on NVIDIA.

Why `answer_v2` despite the 3 lost answers:
- g078 declined arithmetic that no passage states, which is the grounded behaviour wanted;
- g081 rests on an ambiguous question ("primary driver");
- only g012 is a true regression;
- meanwhile hallucination halved (0.114 → 0.057) and citation accuracy rose (0.865 → 0.910).

The prompt is not tuned further on dev, to avoid overfitting; Phase 5's claim verification targets the remaining errors.

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
- **The LLM judge is only partly validated.** Natural answers rarely contain unsupported claims: the hand-labelled samples have 3 and 4. So Cohen's kappa on natural claims is unstable, with confidence intervals spanning most of 0 to 1, and the raw catch count (how many human-flagged unsupported claims the judge also flagged) is reported alongside it. Probe claims are easy cases and overstate agreement on their own.
- **The judge runs at temperature 1.** The judge is OpenAI `gpt-6.1-sol`, which only accepts its default temperature. Judge outputs are cached, so re-running a report reproduces it exactly. A stability check (the same labelled set judged with a second seed) changed 0 of 39 claim verdicts and 0 of 20 answer verdicts.
- **judge_v2's exact-match rule was written after seeing the first sample's labels.** On the fresh second sample it wrongly rejected 7 of 21 supported claims, against 3 of 21 for judge_v1 (labels r2), so judge_v1 remains the prompt in use.
- **The judge is stricter than the labeller on supported claims.** judge_v1 wrongly rejected 2 of 23 natural supported claims in the first sample and 3 of 21 in the fresh one, while catching every claim the labeller marked unsupported (3 of 3 and 4 of 4). Faithfulness and claim hallucination rate are therefore pessimistic: on the fresh sample, about as many of the judge's "unsupported" verdicts were false rejections (3) as true catches (4).
- **Hand labels were corrected after seeing the judge's agreement report.** In both samples, revision r2 changed rows re-checked after the agreement report was shown: 4 rows in the first sample, and in the second only r031, whose re-check was prompted by a judge disagreement. Rows where labeller and judge agreed were not re-checked in the same way, so such corrections tend to move agreement toward the judge. The first label sets are kept, and both are reported.
- **Table chunks can lose their units.** r031's claim misread a value from a table whose "Number in millions" header did not reach the passage, so the generator reported 0.7 million as 0.7%. Table headers and units are a Phase 4 target.
- **The reranker is slow on CPU.** bge-reranker-base via ONNX on CPU adds about 10–20 s per query (20 s with MaxP windows). A GPU or a smaller reranker would be needed before it is worth using live, and on dev it gave no measurable accuracy gain.
- **Evaluation and the live app use different generator providers.** From Phase 4, evaluation generates on NVIDIA NIM's free tier (12–16 s median per answer); the provider check found no change in the primary metrics against Groq, which the live app will use for speed.
- **Phase 4 changes were not measurably better on 44 dev questions.** Headers, the table fix, answer_v2 and confidence-based abstention all have paired CIs that include 0. Small effects would need a larger dev set to detect.
- **Hallucination rate is not shown to improve on test.** It is 0.059 for the baseline and 0.158 for the live config, and no paired difference is real. The baseline's low rate partly reflects its 39% false refusals (0.389 false-abstention rate): an answer that is never given cannot contain unsupported claims.
- **The claim verifier changed little.** On dev it removed one sentence (no over-removal); on test it removed none. It uses the generator model, which may be biased toward its own answers. gpt-oss sometimes returns empty content, so 3 of 32 dev and 1 of 16 test verifications failed and those answers were kept unverified.
- **Small samples.** 44 dev and 23 test questions give wide confidence intervals. Differences between configurations smaller than the CIs shouldn't be read as real.
- **Small, US-focused corpus.** It has three topics and ten documents, mostly from US federal agencies.
