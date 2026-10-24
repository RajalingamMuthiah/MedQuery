# MedQuery eval: `hybrid_rerank_headers_maxp` on the **dev** split

- Run date (UTC): 2026-10-24 05:32
- Run id: `a14cd47892bd`
- Git commit: `cb624f3585c791561c2c72dcd3997046d5e23373` (clean working tree)
- Golden set: `v1` sha256 `ed7ec824b334d36d6be81ab90dc830db78842b414163071a898bc83a39566fb9` (67 items; reviewer: Claude (Anthropic) pre-review against source PDFs)
- Corpus: `data/processed/chunks.jsonl` sha256 `0a57dbba56d6270be48212a8715eda3361096cb957b3978985c4fe6bc76e67f4`

## Config

| setting | value |
|---|---|
| name | `hybrid_rerank_headers_maxp` |
| golden_version | `v1` |
| k | `5` |
| recall_k | `10` |
| embed_model | `BAAI/bge-small-en-v1.5` |
| generator | `openai/gpt-oss-20b` |
| judge | `gpt-6.1-sol` |
| include_quiz | `False` |
| answer_prompt | `answer_v1:9cf9329e3ba3` |
| judge_prompt | `judge_v1:f3afa3a92213` |
| retriever | `hybrid` |
| fuse_k | `60` |
| rerank_model | `BAAI/bge-reranker-base` |
| rerank_n | `20` |
| headers | `True` |
| rerank_window | `320` |
| rerank_overlap | `96` |
| list prices | groq: https://console.groq.com/docs/models (2026-10-21); openai: https://developers.openai.com/api/docs/pricing (2026-10-22) |

## Judge validation

| units | n (probes) | agreement [95% CI] | Cohen's kappa [95% CI] | human / judge |
|---|---|---|---|---|
| all claims | 39 (13) | 0.949 [0.872, 1.000] | 0.896 [0.737, 1.000] | {'supported': 23, 'unsupported': 16} / {'supported': 21, 'unsupported': 18} |
| natural claims | 26 (0) | 0.923 [0.808, 1.000] | 0.708 [0.000, 1.000] | {'supported': 23, 'unsupported': 3} / {'supported': 21, 'unsupported': 5} |
| mismatched-context probes | 13 (13) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | {'unsupported': 13} / {'unsupported': 13} |
| answers | 20 (0) | 0.850 [0.700, 1.000] | 0.615 [0.193, 1.000] | {'correct': 14, 'incorrect': 1, 'partial': 5} / {'correct': 16, 'partial': 4} |

Natural claims: the judge caught **3 of 3** that the labeller marked unsupported, and wrongly rejected **2 of 23** marked supported. With so few unsupported natural claims, kappa is unstable; read it together with these counts.
Labels: revision `r2` by Rajalingam Muthiah (human), sha256 `8d2ee7275bef0ee3d074f44229f17b0daf0d8df86cbce686caa186d641e1a038`; supersedes `judge_review_v1.labels.jsonl` (`ca799c8b9e5373e5f2ebb56cb74001bc2242b81b642aa390b7bed4aab0f4987d`).
Units with no matching re-judged claim (excluded): ['r021'].

### Judge history on the same frozen labels (revision `r2`)

| judge model | prompt | natural unsupported caught | natural supported rejected | probe agreement | answer agreement | answer kappa |
|---|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | `judge_v1:f3afa3a92213` | 1 of 3 | 0 of 24 | 1.000 | 0.850 | 0.634 |
| `gpt-6.1-sol` | `judge_v1:f3afa3a92213` | 3 of 3 | 2 of 23 | 1.000 | 0.850 | 0.615 |
| `gpt-6.1-sol` | `judge_v2:475209e39597` | 3 of 3 | 2 of 23 | 1.000 | 0.850 | 0.615 |
The judge_v2 prompt's exact-match rule was written after seeing these labels, so its measurement here is optimistic; it is re-validated on a fresh sample in Phase 3.

## Items

44 items: 34 answerable, 10 unanswerable. Abstained: 11. Judged: 33. Judge failures: 0. Truncated generations: 0.

Evidence mapping: 44 exact, 0 fuzzy, 0 missing quotes; answerable items with no gold chunk: 0 (excluded from retrieval metrics).

## Metrics (95% bootstrap CI, 1,000 resamples over questions, seed 0)

| metric | value | 95% CI | over | definition |
|---|---|---|---|---|
| answer_correctness | 0.941 | [0.857, 1.000] | answerable | judge vs reference: correct 1, partial 0.5, incorrect 0; abstained = 0 |
| faithfulness | 0.913 | [0.825, 0.981] | answered | per answer, claims supported by retrieved passages / claims; mean |
| hallucination_rate | 0.139 | [0.033, 0.274] | answered | unsupported claims / all claims, pooled |
| answer_relevancy | 1.000 | [1.000, 1.000] | answered | judge rating 1-5 scaled to 0-1 |
| context_precision | 0.631 | [0.518, 0.743] | answerable | rank-aware AP@k over the generator's passages; relevant = maps to a gold quote |
| context_recall | 0.897 | [0.797, 0.974] | answerable | gold evidence items covered by the generator's passages |
| recall@1 | 0.412 | [0.264, 0.557] | answerable | gold chunks in the top 1 / gold |
| recall@3 | 0.809 | [0.682, 0.922] | answerable | gold chunks in the top 3 / gold |
| recall@5 | 0.897 | [0.797, 0.974] | answerable | gold chunks in the top 5 / gold |
| recall@10 | 0.941 | [0.853, 1.000] | answerable | gold chunks in the top 10 / gold |
| mrr | 0.660 | [0.542, 0.770] | answerable | 1 / rank of the first gold chunk in the top 10 |
| citation_accuracy | 0.753 | [0.602, 0.905] | answered | cited claims whose cited passages support them / cited claims, pooled |
| citation_validity | 0.959 | [0.857, 1.000] | answered | citations naming a passage given to the generator / citations |
| uncited_claim_share | 0.076 | [0.000, 0.190] | answered | claims with no citation |
| abstention_accuracy | 0.977 | [0.932, 1.000] | all | abstain on unanswerable, answer on answerable |
| abstention_recall | 1.000 | [1.000, 1.000] | unanswerable | unanswerable questions where the system abstained |
| false_abstention_rate | 0.029 | [0.000, 0.088] | answerable | answerable questions where the system abstained |

Correctness on answerable items: {'abstained': 1, 'correct': 32, 'incorrect': 1}

## Latency and cost

| | p50 | p95 |
|---|---|---|
| retrieval (ms) | 20408.9 | 27302.4 |
| generation (s) | 0.80 | 1.22 |
| end-to-end (s) | 21.12 | 28.22 |

Generation latency is the original API call's, kept in the cache, so re-runs report the same figures. Client-side throttle waits (excluded above): 227.4 s in total.

Cost at list prices: generation $0.000211 per query ($0.0093 total, 112043 tokens); judge $0.2294 total (86762 tokens, evaluation overhead, not part of per-query cost).
