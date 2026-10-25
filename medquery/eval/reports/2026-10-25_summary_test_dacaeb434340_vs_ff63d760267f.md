# MedQuery summary comparison: `hybrid_headers_tables_answer_v2` vs `baseline_nvidia` (test split)

**Summary comparison computed after the final runs; no tuning.** It reads the stored records of two finished runs; nothing was run or changed for it.

- Created (UTC): 2026-10-25 08:02
- Git commit: `150388470b26aa80b851480ad743641b731f9b60` (clean working tree)
- Runs: `baseline_nvidia` `dacaeb434340` and `hybrid_headers_tables_answer_v2` `ff63d760267f`
- Paired bootstrap: 1000 resamples of the same question ids, seed 0; a difference counts as **real** only if its 95% CI excludes 0.

| metric | `baseline_nvidia` | `hybrid_headers_tables_answer_v2` | difference | 95% CI | real? |
|---|---|---|---|---|---|
| answer_correctness | 0.556 | 0.833 | +0.278 | [0.118, 0.469] | **yes** |
| false_abstention_rate | 0.389 | 0.111 | -0.278 | [-0.479, -0.100] | **yes** |
| hallucination_rate | 0.059 | 0.158 | +0.099 | [-0.030, 0.265] | no |
For false_abstention_rate and hallucination_rate a negative difference is an improvement.
