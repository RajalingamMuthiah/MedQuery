# MedQuery ablation: generation quality and the live config (Phase 4) on the **dev** split

- Created (UTC): 2026-10-24 13:39
- Git commit: `cc6b035c9fecebbfa44c99ec0d643ae8dd5db25b` (clean working tree)
- Golden set: `v1` sha256 `ed7ec824b334d36d6be81ab90dc830db78842b414163071a898bc83a39566fb9`; corpus `chunks.jsonl` sha256 v2 `07d7860fac9eae08e3183084722c535b25948e33552979a7161da1686fd90c67`
- Generator `openai/gpt-oss-20b`; answer prompt(s) `answer_v2:0180eca9e1d7`; judge `gpt-6.1-sol` with `judge_v1:f3afa3a92213`; list prices: groq: https://console.groq.com/docs/models (2026-10-21); openai: https://developers.openai.com/api/docs/pricing (2026-10-22); nvidia: https://build.nvidia.com (API catalog free credits, no per-token list price) (2026-10-24)
- Paired bootstrap: 1000 resamples of the same question ids, seed 0; a difference counts as **real** only if its 95% CI excludes 0.
- Phase 5 judge spend so far (all Phase 5 runs): $0.20 of the $3.00 cap
- Citations re-parsed from the stored answers with the current parser for every row.

## Configurations

| row | config | run id | retriever | RRF k | reranker (licence) | rerank N | rerank input | headers | k | generator provider | corpus | answer prompt | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `hybrid_headers_tables_answer_v2` | `53088e55be95` | hybrid | 60 | none | - | - | True | 5 | nvidia | v2 | `answer_v2` |  |
| 2 | `live_verify` | `63e3edbfd540` | hybrid | 60 | none | - | - | True | 5 | nvidia | v2 | `answer_v2` |  |

## Metrics (95% bootstrap CI)

| config | answer_correctness | faithfulness | hallucination_rate | context_recall | recall@5 | recall@10 | mrr | abstention_accuracy | false_abstention_rate | abstention_recall | citation_accuracy | citation_validity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hybrid_headers_tables_answer_v2` | 0.882 [0.781, 0.973] | 0.939 [0.867, 0.991] | 0.057 [0.014, 0.112] | 0.779 [0.632, 0.897] | 0.779 [0.632, 0.897] | 0.897 [0.789, 0.986] | 0.658 [0.522, 0.777] | 0.955 [0.886, 1.000] | 0.059 [0.000, 0.139] | 1.000 [1.000, 1.000] | 0.910 [0.833, 0.965] | 1.000 [1.000, 1.000] |
| `live_verify` | 0.868 [0.767, 0.967] | 0.954 [0.888, 1.000] | 0.043 [0.000, 0.091] | 0.779 [0.632, 0.897] | 0.779 [0.632, 0.897] | 0.897 [0.789, 0.986] | 0.658 [0.522, 0.777] | 0.955 [0.886, 1.000] | 0.059 [0.000, 0.139] | 1.000 [1.000, 1.000] | 0.924 [0.852, 0.981] | 1.000 [1.000, 1.000] |

## Paired differences

Each row against the previous row, except: a k = 8 row against the same config at k = 5, and a MaxP row against the same config with truncated reranker input.

| config vs comparison row | metric | difference | 95% CI | real? |
|---|---|---|---|---|
| `live_verify` vs `hybrid_headers_tables_answer_v2` | answer_correctness | -0.015 | [-0.050, 0.000] | no |
| `live_verify` vs `hybrid_headers_tables_answer_v2` | recall@5 | +0.000 | [0.000, 0.000] | no |
| `live_verify` vs `hybrid_headers_tables_answer_v2` | mrr | +0.000 | [0.000, 0.000] | no |
| `live_verify` vs `hybrid_headers_tables_answer_v2` | false_abstention_rate | +0.000 | [0.000, 0.000] | no |
| `live_verify` vs `hybrid_headers_tables_answer_v2` | hallucination_rate | -0.014 | [-0.049, 0.000] | no |
| `live_verify` vs `hybrid_headers_tables_answer_v2` | abstention_recall | +0.000 | [0.000, 0.000] | no |
| `live_verify` vs `hybrid_headers_tables_answer_v2` | citation_accuracy | +0.014 | [0.000, 0.048] | no |
For false_abstention_rate and hallucination_rate a negative difference is an improvement; for every other metric, a positive one. abstention_recall (unanswerable questions where the system abstained) and citation_accuracy are guards: a real drop in either means better retrieval made the system answer unanswerable questions or cite less accurately.

## Multi-chunk questions

Questions whose answer needs two passages. Context recall counts the gold evidence covered by all k passages the generator saw, so it shows whether a larger k recovers the second gold chunk.

| config | n | answer correctness | context recall | recall@5 | recall@10 | both gold in context |
|---|---|---|---|---|---|---|
| `hybrid_headers_tables_answer_v2` | 5 | 0.800 [0.600, 1.000] | 0.700 [0.300, 1.000] | 0.700 [0.300, 1.000] | 0.900 [0.700, 1.000] | 3 of 5 |
| `live_verify` | 5 | 0.700 [0.500, 0.900] | 0.700 [0.300, 1.000] | 0.700 [0.300, 1.000] | 0.900 [0.700, 1.000] | 3 of 5 |

## Failure diagnosis

Where the best-ranked gold chunk sat, relative to the k passages given to the generator. Misses outside the context are retrieval failures; failures with gold in context point at generation (Phase 4).

| config | false abstentions | incorrect/partial |
|---|---|---|
| `hybrid_headers_tables_answer_v2` | {'not in top 10': 1, 'beyond k': 1} | {'in context': 3} |
| `live_verify` | {'not in top 10': 1, 'beyond k': 1} | {'in context': 3, 'beyond k': 1} |

## Reranker input truncation

Inputs scored are (question, passage) pairs; for a MaxP row, (question, window) pairs. Truncated = over the model's 512-token limit.

| config | inputs scored | over max length (truncated) | share |
|---|---|---|---|
| `hybrid_headers_tables_answer_v2` | - | - | - |
| `live_verify` | - | - | - |

## Latency and cost

| config | retrieval p50 / p95 (ms) | end-to-end p50 / p95 (s) | generation cost per query | judge cost (eval overhead) |
|---|---|---|---|---|
| `hybrid_headers_tables_answer_v2` | 15.2 / 30.8 | 9.77 / 64.94 | $0.000000 | $0.1980 |
| `live_verify` | 16.1 / 35.9 | 31.44 / 110.81 | $0.000000 | $0.1973 |
Retrieval latency is measured live (query embedding, BM25, fusion, rerank); generation latency is the original API call's, from the cache.

## Verifier guards

Over-removal = removed sentences whose matched judge claims (the judge's verdict on the same answer before verification) were all supported. Missed = claims the judge marks unsupported in the verified answers.

| config vs base | answers verified | verifier failures | answers changed | turned into abstentions | sentences removed | removed but judge-supported (over-removal) | removed and judge-unsupported | removed, no judge claim matched | missed |
|---|---|---|---|---|---|---|---|---|---|
| `live_verify` vs `hybrid_headers_tables_answer_v2` | 32 | 3 | 1 | 0 | 1 | 0 | 1 | 0 | 3 |
Verifier latency for `live_verify`: p50 36.39 s, p95 73.06 s per answer (included in end-to-end latency).
