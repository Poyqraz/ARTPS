# Independent Eval V1 — Validation Sanity Audit

Status: **blocked_validation_sanity_review**. Final test remains closed.

Negated-score metrics are diagnostic and cannot be promoted unless the production score contract is objectively demonstrated to have the opposite orientation.

- Protocol: `independent_eval_v1`
- Selected config (historical, immutable): `artps_full_frozen_mars_clf_on_v1`
- Profiles audited: 4
- Objective bug proven: `False`
- Final test authorized: `False`

## Profile summary

| config_id | n | AUROC | AP | thr | CM (tn/fp/fn/tp) | flags |
|---|---:|---:|---:|---:|---|---|
| `artps_full_frozen_raw_clf_on_v1` | 54 | 0.2860 | 0.3822 | 0.0 | 0/27/0/27 | all_positive_predictions, auroc_below_0_5, ap_below_positive_prevalence, class_score_order_reversed, degenerate_threshold_zero |
| `artps_full_frozen_raw_clf_off_v1` | 54 | 0.2750 | 0.3786 | 0.0 | 0/27/0/27 | all_positive_predictions, auroc_below_0_5, ap_below_positive_prevalence, class_score_order_reversed, degenerate_threshold_zero |
| `artps_full_frozen_mars_clf_on_v1` | 54 | 0.3923 | 0.4579 | 0.0 | 0/27/0/27 | all_positive_predictions, auroc_below_0_5, ap_below_positive_prevalence, class_score_order_reversed, degenerate_threshold_zero |
| `artps_full_frozen_mars_clf_off_v1` | 54 | 0.3772 | 0.4490 | 0.0 | 0/27/0/27 | all_positive_predictions, auroc_below_0_5, ap_below_positive_prevalence, class_score_order_reversed, degenerate_threshold_zero |

## Blockers

- `validation_auroc_below_chance`
- `validation_ap_below_balanced_prevalence`
- `degenerate_all_positive_threshold`
- `score_orientation_not_verified`
- `label_score_semantics_not_verified`

## Decision

No objective implementation bug proven from CSV/JSONL cross-checks; below-chance ranking and all-positive threshold=0.0 remain validation sanity blockers. `profile_selection.json` stays immutable. `final_test_authorized=false`; do not open the test split.

