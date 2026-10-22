# MedQuery eval: `baseline` on the **dev** split

- Run date (UTC): 2026-10-22 08:11
- Run id: `eb2f931e39c1`
- Git commit: `fb3f73f6b9a46ccd3d05321cfbfbea2cd6f012cc` (DIRTY working tree)
- Golden set: `v1` sha256 `ed7ec824b334d36d6be81ab90dc830db78842b414163071a898bc83a39566fb9` (67 items; reviewer: Claude (Anthropic) pre-review against source PDFs)
- Corpus: `data/processed/chunks.jsonl` sha256 `0a57dbba56d6270be48212a8715eda3361096cb957b3978985c4fe6bc76e67f4`

## Config

| setting | value |
|---|---|
| name | `baseline` |
| golden_version | `v1` |
| k | `5` |
| recall_k | `10` |
| embed_model | `BAAI/bge-small-en-v1.5` |
| generator | `openai/gpt-oss-20b` |
| judge | `openai/gpt-oss-120b` |
| include_quiz | `False` |
| answer_prompt | `answer_v1:9cf9329e3ba3` |
| judge_prompt | `judge_v1:f3afa3a92213` |
| list prices | https://console.groq.com/docs/models (2026-10-21) |

## Judge validation

**NOT YET VALIDATED.** The judge has not been checked against hand labels (judge_review_v1 pending), so the judge-based metrics (correctness, faithfulness, hallucination rate, relevancy, citation accuracy) are unverified.

## Items

44 items: 34 answerable, 10 unanswerable. Abstained: 17. Judged: 27. Judge failures: 0. Truncated generations: 0.

Evidence mapping: 44 exact, 0 fuzzy, 0 missing quotes; answerable items with no gold chunk: 0 (excluded from retrieval metrics).

## Metrics (95% bootstrap CI, 1,000 resamples over questions, seed 0)

| metric | value | 95% CI | over | definition |
|---|---|---|---|---|
| answer_correctness | 0.706 | [0.545, 0.844] | answerable | judge vs reference: correct 1, partial 0.5, incorrect 0; abstained = 0 |
| faithfulness | 0.981 | [0.940, 1.000] | answered | per answer, claims supported by retrieved passages / claims; mean |
| hallucination_rate | 0.026 | [0.000, 0.083] | answered | unsupported claims / all claims, pooled |
| answer_relevancy | 0.991 | [0.969, 1.000] | answered | judge rating 1-5 scaled to 0-1 |
| context_precision | 0.403 | [0.289, 0.517] | answerable | rank-aware AP@k over the generator's passages; relevant = maps to a gold quote |
| context_recall | 0.676 | [0.515, 0.817] | answerable | gold evidence items covered by the generator's passages |
| recall@1 | 0.250 | [0.113, 0.389] | answerable | gold chunks in the top 1 / gold |
| recall@3 | 0.529 | [0.371, 0.689] | answerable | gold chunks in the top 3 / gold |
| recall@5 | 0.662 | [0.513, 0.797] | answerable | gold chunks in the top 5 / gold |
| recall@10 | 0.779 | [0.647, 0.910] | answerable | gold chunks in the top 10 / gold |
| mrr | 0.481 | [0.348, 0.608] | answerable | 1 / rank of the first gold chunk in the top 10 |
| citation_accuracy | 0.949 | [0.867, 1.000] | answered | cited claims whose cited passages support them / cited claims, pooled |
| citation_validity | 0.972 | [0.902, 1.000] | answered | citations naming a passage given to the generator / citations |
| uncited_claim_share | 0.000 | [0.000, 0.000] | answered | claims with no citation |
| abstention_accuracy | 0.841 | [0.727, 0.932] | all | abstain on unanswerable, answer on answerable |
| abstention_recall | 1.000 | [1.000, 1.000] | unanswerable | unanswerable questions where the system abstained |
| false_abstention_rate | 0.206 | [0.086, 0.364] | answerable | answerable questions where the system abstained |

Correctness on answerable items: {'abstained': 7, 'correct': 23, 'incorrect': 2, 'partial': 2}

## Latency and cost

| | p50 | p95 |
|---|---|---|
| retrieval (ms) | 13.5 | 21.5 |
| generation (s) | 0.68 | 1.10 |
| end-to-end (s) | 0.70 | 1.11 |

Generation latency is the original API call's, kept in the cache, so re-runs report the same figures. Client-side throttle waits (excluded above): 407.4 s in total.

Cost at list prices: generation $0.000196 per query ($0.0086 total, 104156 tokens); judge $0.0149 total (71831 tokens, evaluation overhead, not part of per-query cost).
