# MedQuery eval: `baseline` on the **test** split

- Run date (UTC): 2026-10-22 10:07
- Run id: `b2bcda7e0fda`
- Git commit: `8e23ccffdc0d4ad605a68a53c792fdfeb6d9c5c2` (clean working tree)
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
| judge | `gpt-6.1-sol` |
| include_quiz | `False` |
| answer_prompt | `answer_v1:9cf9329e3ba3` |
| judge_prompt | `judge_v1:f3afa3a92213` |
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

23 items: 18 answerable, 5 unanswerable. Abstained: 12. Judged: 11. Judge failures: 0. Truncated generations: 0.

Evidence mapping: 27 exact, 0 fuzzy, 0 missing quotes; answerable items with no gold chunk: 0 (excluded from retrieval metrics).

## Metrics (95% bootstrap CI, 1,000 resamples over questions, seed 0)

| metric | value | 95% CI | over | definition |
|---|---|---|---|---|
| answer_correctness | 0.611 | [0.368, 0.812] | answerable | judge vs reference: correct 1, partial 0.5, incorrect 0; abstained = 0 |
| faithfulness | 0.909 | [0.700, 1.000] | answered | per answer, claims supported by retrieved passages / claims; mean |
| hallucination_rate | 0.033 | [0.000, 0.137] | answered | unsupported claims / all claims, pooled |
| answer_relevancy | 1.000 | [1.000, 1.000] | answered | judge rating 1-5 scaled to 0-1 |
| context_precision | 0.335 | [0.173, 0.516] | answerable | rank-aware AP@k over the generator's passages; relevant = maps to a gold quote |
| context_recall | 0.611 | [0.406, 0.824] | answerable | gold evidence items covered by the generator's passages |
| recall@1 | 0.222 | [0.053, 0.438] | answerable | gold chunks in the top 1 / gold |
| recall@3 | 0.333 | [0.133, 0.571] | answerable | gold chunks in the top 3 / gold |
| recall@5 | 0.611 | [0.406, 0.824] | answerable | gold chunks in the top 5 / gold |
| recall@10 | 0.750 | [0.556, 0.917] | answerable | gold chunks in the top 10 / gold |
| mrr | 0.373 | [0.226, 0.547] | answerable | 1 / rank of the first gold chunk in the top 10 |
| citation_accuracy | 0.967 | [0.863, 1.000] | answered | cited claims whose cited passages support them / cited claims, pooled |
| citation_validity | 1.000 | [1.000, 1.000] | answered | citations naming a passage given to the generator / citations |
| uncited_claim_share | 0.000 | [0.000, 0.000] | answered | claims with no citation |
| abstention_accuracy | 0.696 | [0.521, 0.870] | all | abstain on unanswerable, answer on answerable |
| abstention_recall | 1.000 | [1.000, 1.000] | unanswerable | unanswerable questions where the system abstained |
| false_abstention_rate | 0.389 | [0.188, 0.632] | answerable | answerable questions where the system abstained |

Correctness on answerable items: {'abstained': 7, 'correct': 11}

## Latency and cost

| | p50 | p95 |
|---|---|---|
| retrieval (ms) | 9.8 | 15.9 |
| generation (s) | 0.73 | 1.61 |
| end-to-end (s) | 0.75 | 1.62 |

Generation latency is the original API call's, kept in the cache, so re-runs report the same figures. Client-side throttle waits (excluded above): 539.3 s in total.

Cost at list prices: generation $0.000184 per query ($0.0042 total, 51828 tokens); judge $0.0688 total (24362 tokens, evaluation overhead, not part of per-query cost).
