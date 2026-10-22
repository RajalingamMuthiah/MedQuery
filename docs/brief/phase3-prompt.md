Phase 2 is approved. Update CLAUDE.md, then propose the Phase 3 plan: retrieval upgrades. Plan only; no code until I approve.

## 0. Diagnose first (dev only, from the existing dev run; no API calls)
For each dev false abstention and each incorrect/partial answer: was a gold chunk in the top 5, in ranks 6-10, or not in the top 10? Report the counts. This separates retrieval misses (Phase 3) from generator over-caution (Phase 4).

## 1. Configurations (one ablation row each, added cumulatively)
- baseline (existing)
- + BM25 hybrid with reciprocal rank fusion (justify the RRF constant; no tuning on test)
- + cross-encoder reranker over the fused top-N (CPU; prefer an ONNX/no-torch option such as fastembed's reranker if available; justify N and the model; record its licence)
- + contextual chunk headers (prepend title and section path to the text that is embedded and indexed, not to what the generator is shown) — tested as its own row
- optionally k=5 vs k=8 for generation, only if the diagnosis shows gold chunks often sit at ranks 6-10
Generator, prompt (answer_v1), judge (gpt-6.1-sol, judge_v1) and golden_v1 stay fixed so rows are comparable.

## 2. Comparison method
- Dev split only for every Phase 3 decision. Do NOT run the test split in Phase 3.
- For each config vs the previous row: paired bootstrap on the same questions (1,000 resamples, seed 0) for the difference in answer correctness, recall@5, MRR and false-abstention rate, with 95% CI. Call an improvement real only if the difference CI excludes 0.
- An ablation table in a report (config, metric, value, paired difference vs previous, CI), plus latency p50/p95 and cost per query for each row (the reranker adds latency).

## 3. Fresh judge re-validation
Sample a NEW judge-review set (about 25 claims including at least 5 judge-unsupported, plus 10 answers) from the best Phase 3 dev run, excluding any item/claim already in judge_review_v1. Verdicts hidden as before. STOP for my hand labels, then report agreement for gpt-6.1-sol judge_v1 on this independent sample. This validates the judge on data its prompt wasn't tuned on.

## 4. Rules
- Offline tests for BM25, RRF, the reranker wrapper (fake model) and the paired bootstrap.
- Cache everything; resumable runs; reports from clean commits.
- Estimate API cost and time per dev run before running.
- Logical commits, push, CI green, CLAUDE.md and README (ablation table from reports only), summary, STOP.

Ask me up to 3 questions if anything conflicts with the repo.
