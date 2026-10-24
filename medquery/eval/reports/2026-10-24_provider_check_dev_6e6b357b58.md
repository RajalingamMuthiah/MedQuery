# MedQuery provider check: generator on Groq vs NVIDIA NIM (dev split)

- Created (UTC): 2026-10-24 09:18
- Git commit: `94439dd9447d25c1ab9bc27c049d2592606fd91a` (clean working tree)
- Golden set: `v1` sha256 `ed7ec824b334d36d6be81ab90dc830db78842b414163071a898bc83a39566fb9`; corpus `chunks.jsonl` sha256 `v1 `0a57dbba56d6270be48212a8715eda3361096cb957b3978985c4fe6bc76e67f4``
- Same model `openai/gpt-oss-20b`, prompt `answer_v1:9cf9329e3ba3`, judge `gpt-6.1-sol` with `judge_v1:f3afa3a92213`. Retrieval is identical within each pair, so differences come from generation only.
- Paired bootstrap: 1000 resamples of the same question ids, seed 0; a difference counts as **real** only if its 95% CI excludes 0.
- Citations are re-parsed from the stored answers with the current parser (Unicode hyphens and fullwidth brackets normalised) for both providers.
- Prices: groq: https://console.groq.com/docs/models (2026-10-21); openai: https://developers.openai.com/api/docs/pricing (2026-10-22); nvidia: https://build.nvidia.com (API catalog free credits, no per-token list price) (2026-10-24)

## `hybrid`: Groq run `d17f758c5f0c` vs NVIDIA run `ed1249bdc055`

| metric | Groq | NVIDIA | NVIDIA - Groq | 95% CI | real? |
|---|---|---|---|---|---|
| answer_correctness | 0.912 | 0.912 | +0.000 | [0.000, 0.000] | no |
| faithfulness | 0.924 | 0.935 | +0.010 | [0.000, 0.036] | no |
| hallucination_rate | 0.076 | 0.059 | -0.017 | [-0.056, 0.004] | no |
| context_recall | 0.779 | 0.779 | +0.000 | [0.000, 0.000] | no |
| recall@5 | 0.779 | 0.779 | +0.000 | [0.000, 0.000] | no |
| recall@10 | 0.824 | 0.824 | +0.000 | [0.000, 0.000] | no |
| mrr | 0.533 | 0.533 | +0.000 | [0.000, 0.000] | no |
| abstention_accuracy | 0.955 | 0.955 | +0.000 | [0.000, 0.000] | no |
| false_abstention_rate | 0.059 | 0.059 | +0.000 | [0.000, 0.000] | no |
| abstention_recall | 1.000 | 1.000 | +0.000 | [0.000, 0.000] | no |
| citation_accuracy | 0.803 | 0.873 | +0.070 | [-0.013, 0.158] | no |
| citation_validity | 0.975 | 1.000 | +0.025 | [0.000, 0.086] | no |

| | Groq | NVIDIA |
|---|---|---|
| generation latency p50 / p95 (s) | 0.72 / 1.12 | 12.12 / 58.06 |
| truncated generations | 0 | 0 |
| generation cost per query | $0.000207 | $0.000000 |
| judge cost (eval overhead) | $0.2100 | $0.2089 |

## `hybrid_rerank_headers_maxp`: Groq run `a14cd47892bd` vs NVIDIA run `4b3e118c95f5`

| metric | Groq | NVIDIA | NVIDIA - Groq | 95% CI | real? |
|---|---|---|---|---|---|
| answer_correctness | 0.941 | 0.912 | -0.029 | [-0.091, 0.000] | no |
| faithfulness | 0.913 | 0.930 | +0.016 | [0.000, 0.036] | **yes** |
| hallucination_rate | 0.139 | 0.083 | -0.056 | [-0.111, 0.001] | no |
| context_recall | 0.897 | 0.897 | +0.000 | [0.000, 0.000] | no |
| recall@5 | 0.897 | 0.897 | +0.000 | [0.000, 0.000] | no |
| recall@10 | 0.941 | 0.941 | +0.000 | [0.000, 0.000] | no |
| mrr | 0.660 | 0.660 | +0.000 | [0.000, 0.000] | no |
| abstention_accuracy | 0.977 | 0.955 | -0.023 | [-0.068, 0.000] | no |
| false_abstention_rate | 0.029 | 0.059 | +0.029 | [0.000, 0.091] | no |
| abstention_recall | 1.000 | 1.000 | +0.000 | [0.000, 0.000] | no |
| citation_accuracy | 0.753 | 0.825 | +0.072 | [-0.011, 0.161] | no |
| citation_validity | 1.000 | 1.000 | +0.000 | [0.000, 0.000] | no |

| | Groq | NVIDIA |
|---|---|---|
| generation latency p50 / p95 (s) | 0.80 / 1.22 | 15.74 / 71.73 |
| truncated generations | 0 | 0 |
| generation cost per query | $0.000211 | $0.000000 |
| judge cost (eval overhead) | $0.2294 | $0.2163 |

**Verdict:** at least one paired difference is real: the provider switch changes results.
