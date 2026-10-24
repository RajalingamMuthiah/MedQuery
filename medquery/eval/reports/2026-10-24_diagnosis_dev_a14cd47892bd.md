# MedQuery failure diagnosis: run `a14cd47892bd` (`hybrid_rerank_headers_maxp`, dev)

- Created (UTC): 2026-10-24 08:18
- Git commit: `e803080bb4dfd1abd0e7ff5b3cc1dd6809c9e4b1` (clean working tree)
- Causes: hand-assigned in `medquery/eval/diagnosis/phase4_causes.csv`; the findings come from the run's judge verdicts. No API calls.

## Findings by cause

| cause | findings | items |
|---|---|---|
| table/units misread | 2 | g018 |
| number/qualifier near-miss | 1 | g075 |
| multi-passage synthesis | 1 | g022 |
| citation pointing to the wrong passage | 1 | g077 |
| other: detail not in the passages | 6 | g092 |
| other: paraphrase beyond the passage | 1 | g001 |
| retrieval: gold not in the top 10 | 1 | g031 |

## Every finding

| item | type | finding | claim | cause | note |
|---|---|---|---|---|---|
| g001 | factual | unsupported_claim | Insulin is needed to move glucose from the blood into cells for energy. | other: paraphrase beyond the passage | Mechanism paraphrased beyond the passage wording; hand label: supported (judge_review_v2 r008). |
| g018 | table | incorrect | The estimated percentage of undiagnosed diabetes among non‑Hispanic Asian adults is 0.7% (95% CI 0.4–1.0)【dm-stats-2020:006】 | table/units misread | Asked for Table 1a, which was not among the 5 passages; the answer read Table 1b and reported 0.7 million as 0.7%. It also wrote a malformed citation (fullwidth brackets, 3-digit id). |
| g018 | table | unsupported_claim | The estimated percentage of undiagnosed diabetes among non-Hispanic Asian adults is 0.7% (95% CI 0.4–1.0). | table/units misread | 0.7 (0.4-1.0) is in the 'Number in Millions' column of Table 1b, not a percentage. |
| g022 | multi-chunk | unsupported_claim | In 2013–2016, 89.0% of adults with diagnosed diabetes were overweight or had obesity, defined as BMI at or above 25 kg/m². | multi-passage synthesis | The population and period come from the other cited passage; hand label: supported (judge_review_v2 r030). |
| g031 | factual | false_abstention | Among individuals flagged for possible undetected high blood pressure who came back for a check-up, what proportion was eventually diagnosed with hypertension? | retrieval: gold not in the top 10 | The gold chunk (htn-millionhearts:0014) is not in the top 10. |
| g075 | factual | unsupported_claim | Patients should keep this documentation for future medical encounters. | number/qualifier near-miss | 'Future TB testing' became 'future medical encounters'; hand label: supported (judge_review_v2 r009). |
| g077 | table | unsupported_claim | Staff and visitors should use respiratory protection such as an N95 or higher filtering facepiece respirator when in close proximity to the resident. | citation pointing to the wrong passage | The claim is supported by another passage in the context but cites tb-core-curriculum:0316. |
| g092 | superseded | unsupported_claim | Weight loss was a recommended lifestyle modification for individuals with prehypertension. | other: detail not in the passages | Weight loss: none of the 5 passages lists the specific lifestyle modifications; likely model knowledge. The answer also omits the superseded warning (answer_v1 rule 5). |
| g092 | superseded | unsupported_claim | Reduced dietary sodium was recommended for individuals with prehypertension. | other: detail not in the passages | Reduced sodium: not in any of the 5 passages. |
| g092 | superseded | unsupported_claim | Increased dietary potassium was recommended for individuals with prehypertension. | other: detail not in the passages | Increased potassium: not in any of the 5 passages. |
| g092 | superseded | unsupported_claim | Regular aerobic exercise was recommended for individuals with prehypertension. | other: detail not in the passages | Regular aerobic exercise: not in any of the 5 passages. |
| g092 | superseded | unsupported_claim | Moderation of alcohol intake was recommended for individuals with prehypertension. | other: detail not in the passages | Moderation of alcohol intake: not in any of the 5 passages. |
| g092 | superseded | unsupported_claim | Smoking cessation was recommended for individuals with prehypertension. | other: detail not in the passages | Smoking cessation: not in any of the 5 passages. |

## Table chunks (corpus v1)

| measure | chunks |
|---|---|
| chunks | 1133 |
| table chunks | 207 |
| no caption | 137 |
| caption cut mid sentence | 66 |
| continuation without caption or header | 50 |
| multi row header | 93 |
| all empty columns | 83 |
| split decimals | 8 |

Continuation = a captionless table on the page after a table of the same document with the same number of columns. The table_v2 fix targets every row of this table.
