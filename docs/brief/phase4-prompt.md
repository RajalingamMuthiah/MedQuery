Phase 4: generation quality and the live config. Plan only; no code until I approve. Dev split only; do NOT run the test split (the final test runs happen once, at the end of Phase 5).

## 0. Provider switch (do first)
Move the generator to NVIDIA NIM, same model openai/gpt-oss-20b (NVIDIA_API_KEY is in .env). Add an "nvidia" provider behind the existing LLM interface (same throttle/retry/cache/redaction; key never printed or committed; add NVIDIA_API_KEY= to .env.example). Probe the exact model id and which params are accepted (temperature 0, seed, reasoning_effort, max_tokens); mirror the Groq settings and record differences. Make the generator provider part of RunConfig and the run id. Re-run two rows on NVIDIA as Phase 4's starting points: hybrid and hybrid_rerank_headers_maxp. Report a "provider check" (Groq vs NVIDIA on those rows, paired differences).

## 1. Diagnose first (no API calls, existing dev runs)
- Every incorrect/partial answer and every claim the judge marked unsupported in the best Phase 3 run: classify the cause (table/units misread, number/qualifier near-miss, multi-passage synthesis, citation pointing to the wrong passage, other).
- Table chunks: how many lose their column headers or units (e.g. Table 1b "Number in millions" shows as empty header cells)? This caused a real error (0.7 million read as 0.7%).

## 2. Candidate changes (one ablation row each, cumulative, on NVIDIA)
- hybrid_headers: hybrid + headers WITHOUT the reranker (the likely live config; the reranker added 10–20 s/query on CPU with no measurable gain).
- Table fix: carry each table's caption/title and column headers (including units) into every table chunk; rebuild as a new corpus version; never mutate the old one.
- answer_v2 prompt: cite every sentence with the passage it came from; copy numbers, units and qualifiers exactly as written; if a needed fact is missing, say which part cannot be answered instead of guessing; keep the superseded-guidance warning. Version it, hash it, keep answer_v1 for comparison.
- Abstention by retrieval confidence: compute a risk–coverage curve offline from existing records (accuracy vs share of questions answered, thresholding a retrieval score), and report the operating point; no new API calls needed for the curve.
Each row: paired bootstrap vs the previous row (1,000 resamples, seed 0), real only if the CI excludes 0. Guards: abstention_recall and citation_accuracy. Report latency p50/p95.

## 3. Budget safety
Before each run, estimate the OpenAI judge cost and refuse to start if the phase total would exceed $3. Print the running total in every report.

## 4. Rules
Offline tests for the provider, table-chunk rebuild, prompt versioning and the risk–coverage computation. Reports from clean commits only. Logical commits, push, CI green, README (from reports only), CLAUDE.md, summary, STOP.

Ask me up to 3 questions if anything conflicts with the repo.
