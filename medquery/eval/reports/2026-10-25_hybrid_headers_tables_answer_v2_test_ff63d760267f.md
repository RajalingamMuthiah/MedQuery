# MedQuery eval: `hybrid_headers_tables_answer_v2` on the **test** split

- Run date (UTC): 2026-10-25 05:25
- Run id: `ff63d760267f`
- Git commit: `a28590b100903bc28018867b3b079d09f336911d` (clean working tree)
- Golden set: `v1` sha256 `ed7ec824b334d36d6be81ab90dc830db78842b414163071a898bc83a39566fb9` (67 items; reviewer: Claude (Anthropic) pre-review against source PDFs)
- Corpus: `data/processed/chunks.jsonl` sha256 `07d7860fac9eae08e3183084722c535b25948e33552979a7161da1686fd90c67`
- Phase 5 judge spend so far (all Phase 5 runs, this one included): $0.45 of the $3.00 cap

## Config

| setting | value |
|---|---|
| name | `hybrid_headers_tables_answer_v2` |
| golden_version | `v1` |
| k | `5` |
| recall_k | `10` |
| embed_model | `BAAI/bge-small-en-v1.5` |
| generator | `openai/gpt-oss-20b` |
| judge | `gpt-6.1-sol` |
| include_quiz | `False` |
| answer_prompt | `answer_v2:0180eca9e1d7` |
| judge_prompt | `judge_v1:f3afa3a92213` |
| retriever | `hybrid` |
| fuse_k | `60` |
| rerank_model | `` |
| rerank_n | `20` |
| headers | `True` |
| rerank_window | `0` |
| rerank_overlap | `0` |
| generator_provider | `nvidia` |
| corpus | `v2` |
| verifier | `` |
| list prices | groq: https://console.groq.com/docs/models (2026-10-21); openai: https://developers.openai.com/api/docs/pricing (2026-10-22); nvidia: https://build.nvidia.com (API catalog free credits, no per-token list price) (2026-10-24) |

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

23 items: 18 answerable, 5 unanswerable. Abstained: 7. Judged: 16. Judge failures: 0. Truncated generations: 0.

Evidence mapping: 27 exact, 0 fuzzy, 0 missing quotes; answerable items with no gold chunk: 0 (excluded from retrieval metrics).

## Metrics (95% bootstrap CI, 1,000 resamples over questions, seed 0)

| metric | value | 95% CI | over | definition |
|---|---|---|---|---|
| answer_correctness | 0.833 | [0.656, 0.972] | answerable | judge vs reference: correct 1, partial 0.5, incorrect 0; abstained = 0 |
| faithfulness | 0.818 | [0.625, 0.969] | answered | per answer, claims supported by retrieved passages / claims; mean |
| hallucination_rate | 0.158 | [0.030, 0.346] | answered | unsupported claims / all claims, pooled |
| answer_relevancy | 0.984 | [0.950, 1.000] | answered | judge rating 1-5 scaled to 0-1 |
| context_precision | 0.560 | [0.377, 0.735] | answerable | rank-aware AP@k over the generator's passages; relevant = maps to a gold quote |
| context_recall | 0.750 | [0.528, 0.932] | answerable | gold evidence items covered by the generator's passages |
| recall@1 | 0.389 | [0.188, 0.611] | answerable | gold chunks in the top 1 / gold |
| recall@3 | 0.750 | [0.528, 0.932] | answerable | gold chunks in the top 3 / gold |
| recall@5 | 0.750 | [0.528, 0.932] | answerable | gold chunks in the top 5 / gold |
| recall@10 | 0.833 | [0.658, 0.972] | answerable | gold chunks in the top 10 / gold |
| mrr | 0.590 | [0.423, 0.746] | answerable | 1 / rank of the first gold chunk in the top 10 |
| citation_accuracy | 0.842 | [0.654, 0.970] | answered | cited claims whose cited passages support them / cited claims, pooled |
| citation_validity | 1.000 | [1.000, 1.000] | answered | citations naming a passage given to the generator / citations |
| uncited_claim_share | 0.000 | [0.000, 0.000] | answered | claims with no citation |
| abstention_accuracy | 0.913 | [0.783, 1.000] | all | abstain on unanswerable, answer on answerable |
| abstention_recall | 1.000 | [1.000, 1.000] | unanswerable | unanswerable questions where the system abstained |
| false_abstention_rate | 0.111 | [0.000, 0.286] | answerable | answerable questions where the system abstained |

Correctness on answerable items: {'abstained': 2, 'correct': 14, 'partial': 2}

## Latency and cost

| | p50 | p95 |
|---|---|---|
| retrieval (ms) | 19.6 | 35.5 |
| generation (s) | 1.93 | 5.68 |
| end-to-end (s) | 1.94 | 5.70 |

Generation latency is the original API call's, kept in the cache, so re-runs report the same figures. Client-side throttle waits (excluded above): 0.0 s in total.

Cost at list prices: generation $0.000000 per query ($0.0000 total, 55704 tokens); judge $0.0951 total (34593 tokens, evaluation overhead, not part of per-query cost).
