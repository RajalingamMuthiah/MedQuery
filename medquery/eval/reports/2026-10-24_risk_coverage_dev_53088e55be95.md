# MedQuery risk-coverage: run `53088e55be95` (`hybrid_headers_tables_answer_v2`, dev)

- Created (UTC): 2026-10-24 11:09
- Git commit: `c0fdfd0c0ec4398faef8397035126a6f11902ad5` (clean working tree)
- Signal: top-1 dense cosine similarity between the question and the passages given to the generator (local embeddings, no API calls). An answer below the threshold abstains.
- Score: answerable = correctness (1 / 0.5 / 0; abstaining = 0); unanswerable = 1 if the system abstains. Overall = mean over all questions.
- Operating point rule (fixed before looking): maximise the overall dev score; ties go to the lower threshold.

## Operating point

| | value |
|---|---|
| threshold | none (answer all) |
| overall score, no threshold | 0.909 |
| overall score at the threshold (in-sample, optimistic) | 0.909 |
| overall score, leave-one-out (threshold chosen on the other questions) | 0.909 |
| answers turned into abstentions | 0 |
| leave-one-out thresholds | None to None (44 folds chose none) |

## Risk-coverage curve

| threshold | coverage | accuracy among answered | overall |
|---|---|---|---|
| none | 0.727 | 0.938 | 0.909 |
| 0.6771 | 0.727 | 0.938 | 0.909 |
| 0.7157 | 0.727 | 0.938 | 0.909 |
| 0.7380 | 0.705 | 0.935 | 0.886 |
| 0.7388 | 0.682 | 0.933 | 0.864 |
| 0.7407 | 0.682 | 0.933 | 0.864 |
| 0.7411 | 0.682 | 0.933 | 0.864 |
| 0.7422 | 0.659 | 0.931 | 0.841 |
| 0.7470 | 0.636 | 0.929 | 0.818 |
| 0.7542 | 0.614 | 0.926 | 0.795 |
| 0.7570 | 0.591 | 0.923 | 0.773 |
| 0.7667 | 0.568 | 0.920 | 0.750 |
| 0.7791 | 0.545 | 0.917 | 0.727 |
| 0.7861 | 0.523 | 0.913 | 0.705 |
| 0.7878 | 0.523 | 0.913 | 0.705 |
| 0.7916 | 0.500 | 0.932 | 0.693 |
| 0.7928 | 0.500 | 0.932 | 0.693 |
| 0.7934 | 0.500 | 0.932 | 0.693 |
| 0.7962 | 0.477 | 0.929 | 0.670 |
| 0.7996 | 0.455 | 0.925 | 0.648 |
| 0.8048 | 0.432 | 0.921 | 0.625 |
| 0.8048 | 0.432 | 0.921 | 0.625 |
| 0.8082 | 0.409 | 0.917 | 0.602 |
| 0.8107 | 0.409 | 0.917 | 0.602 |
| 0.8142 | 0.386 | 0.912 | 0.580 |
| 0.8163 | 0.364 | 0.906 | 0.557 |
| 0.8168 | 0.341 | 0.900 | 0.534 |
| 0.8204 | 0.341 | 0.900 | 0.534 |
| 0.8230 | 0.341 | 0.900 | 0.534 |
| 0.8231 | 0.318 | 0.893 | 0.511 |
| 0.8258 | 0.295 | 0.885 | 0.489 |
| 0.8328 | 0.295 | 0.885 | 0.489 |
| 0.8339 | 0.273 | 0.875 | 0.466 |
| 0.8441 | 0.250 | 0.955 | 0.466 |
| 0.8468 | 0.227 | 0.950 | 0.443 |
| 0.8472 | 0.205 | 0.944 | 0.420 |
| 0.8475 | 0.182 | 0.938 | 0.398 |
| 0.8502 | 0.159 | 1.000 | 0.386 |
| 0.8555 | 0.136 | 1.000 | 0.364 |
| 0.8564 | 0.114 | 1.000 | 0.341 |
| 0.8584 | 0.091 | 1.000 | 0.318 |
| 0.8617 | 0.068 | 1.000 | 0.295 |
| 0.8843 | 0.045 | 1.000 | 0.273 |
| 0.8870 | 0.023 | 1.000 | 0.250 |
| 0.8933 | 0.000 | n/a | 0.227 |
