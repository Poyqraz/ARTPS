# independent_eval_v1_1 — full human-reviewed annotation + frozen validation audit

status: complete
review_type: repeat_author_blind_review
independent_annotator: false
reviewed: 360/360
annotation_version: independent_eval_v1_1
evaluation_name: independent_eval_v1_1 evaluation
not_a_historical_paper_correction: true

This is a **current reproducible evaluation** of the supplementary
`independent_eval_v1` benchmark under a new human-reviewed annotation version.
It is **not** a correction of the historical ARTPS manuscript / accepted IAC
numbers (AUROC 0.894, AUPRC 0.847, F1 0.823, baseline AUROC 0.856, 28.1 FPS).
Those values and C05/C06/C07 `accepted_abstract_reproduction_pending` status
are unchanged.

## 306 remaining review (progress gate)

- n: 306/306 (`review_0055`..`review_0360`)
- canonical: positive/1 = 294, negative/0 = 12, uncertain = 0, exclude = 0
- confidence: high = 291, medium = 3, low = 12
- reviewer_role: `repeat_author_review` (all)
- model_blind / existing_label_hidden / terrain_hidden / source_partition_hidden: true
- independent_annotator: false (not a second independent annotation)
- test images were shown for human annotation only; **no ARTPS inference** on test

## Full 360 original-vs-review comparison (label-only)

Model scores were not included.

| original \ review | review 0 | review 1 |
|---|---|---|
| original 0 | 14 | 166 |
| original 1 | 6 | 174 |

- n_reviewed: 360
- agreement: 188 (0.522)
- disagreement: 172 (0.478)
- uncertain: 0 / exclude: 0
- original_negative_to_review_positive: 166
- original_positive_to_review_negative: 6
- confidence: high 329 / medium 14 / low 17

Per-split disagreement (post-completion only): train 122/252, validation 25/54, test 25/54.

`independent_eval_v1` was **not** overwritten.

## Frozen `independent_eval_v1_1` distribution (no forced balance)

Overall final_label: **340 positive / 20 negative** (emerged from review; not capped).

| split | positive | negative |
|---|---|---|
| train | 242 | 10 |
| validation | 46 | 8 |
| test | 52 | 2 |

- split assignments unchanged vs v1
- no single-class split → `independent_eval_v1_1_split_v2` **not** proposed
- duplicate/scene/SHA/source/sequence leakage: none
- uncertain/exclude auto-binarized: false

## Frozen ARTPS validation remetrics (FP32, no inference rerun)

Profile: `artps_full_frozen_mars_clf_on_v1` (same frozen scores as v1 selection).
Only `y_true` was remapped to v1_1 human labels. `profile_selection.json` was not mutated.

| | v1 (heuristic labels) | v1_1 (human-reviewed labels) |
|---|---|---|
| AUROC | 0.392 | 0.772 |
| AP | 0.458 | 0.956 |
| F1 @ selected threshold | 0.667 | 0.920 |
| selected threshold | 0.0 | 0.0 |
| confusion (tn/fp/fn/tp) | 0/27/0/27 | 0/8/0/46 |
| validation n (pos/neg) | 27/27 | 46/8 |

The selected threshold remains **0.0** (all-positive operating point) under the existing
validation-F1 / highest-threshold tie-break policy. F1 rose largely because prevalence
changed (46/8 vs 27/27), not because a new non-degenerate operating point was found.
AUROC/AP are ranking metrics on the same frozen scores against the new labels.

Do **not** auto-select the numerically higher row as “the” result. Report both as an audit:
v1 = heuristic-label evaluation; v1_1 = human-reviewed-label evaluation.

## Final test

recommendation: **keep_closed**

Reasons: degenerate threshold still 0.0; extreme class imbalance (test negatives = 2);
repeat-author review is not an independent second annotation; test embargo remains the
protocol default until the user explicitly authorizes.

- `test_opened: false`
- `final_test_authorized: false`
- `FINAL_TEST_SCOPE.status: blocked_validation_sanity_review`

## Invariants (unchanged)

- Historical manuscript claims unchanged
- C05/C06/C07 stay `accepted_abstract_reproduction_pending`
- No ARTPS tuning / threshold / fusion / mask / preprocessing / orientation / profile change
- No test predictions or test metrics produced
