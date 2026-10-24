Phase 5: claim-level verification, then the final test-split results. Plan only; no code until I approve.

## 1. Verifier (dev only)
After answer_v2 generates, split the answer into atomic claims, check each claim against the cited/context passages, and remove or flag unsupported claims (re-render the answer from supported claims; if nothing survives, abstain).
- The verifier must NOT be the judge model (gpt-6.1-sol), so the evaluation stays independent. Propose options with trade-offs: (a) a local CPU NLI/entailment model (check licence, size, latency), (b) the generator model on NVIDIA/Groq with a verification prompt. Recommend one.
- Measure on dev as one new row after the live config: answer correctness, hallucination rate, faithfulness, citation accuracy, false-abstention rate, abstention recall, latency p50/p95. Paired bootstrap vs the live config. Stop rule uses primary metrics only (answer correctness, false-abstention rate, hallucination rate).
- Guard: report how many claims the verifier removed that the judge marked supported (over-removal), and how many unsupported claims it missed.

## 2. Final test-split runs (once, at the end)
After the verifier decision, from a clean commit, run on the TEST split exactly once each: baseline (Phase 2), hybrid (Phase 3), the live config, and live config + verifier (if adopted). No tuning after seeing test results. Final ablation table on test with 95% CIs and paired differences, clearly marked as the headline results; dev results stay as development history.

## 3. Rules
Budget cap for the OpenAI judge this phase: $3, enforced before each run. Offline tests for the verifier (fake model), claim splitting and answer re-rendering. Reports from clean commits. README: a "Final results (test split)" section at the top, from reports only, with limitations. CLAUDE.md, logical commits, push, CI green, summary, STOP.

Ask me up to 3 questions if anything conflicts with the repo.
