# Independent Eval V1 — Validation Sanity Audit

Status: **blocked_validation_sanity_review**. Final test remains closed.

Negated-score metrics are diagnostic and cannot be promoted unless the production score contract is objectively demonstrated to have the opposite orientation.

- Protocol: `independent_eval_v1`
- Selected config (historical, immutable): `artps_full_frozen_mars_clf_on_v1`
- Profiles audited: 4
- Metric bug detected: `False`
- Label mapping bug detected: `False`
- Final test authorized: `False`

## Scoped verification flags

- `metric_bug_detected`: `False`
- `label_mapping_bug_detected`: `False`
- `duplicate_prediction_bug_detected`: `False`
- `processing_error_mass_failure_detected`: `False`
- `score_semantics_fully_verified`: `False`
- `classifier_class_semantics_verified`: `False`
- `candidate_suppression_semantics_verified`: `False`
- `annotation_quality_independently_verified`: `False`

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
- `blind_review_pending`
- `classifier_class_semantics_unverified`
- `candidate_suppression_semantics_unverified`

## Decision

No metric-computation or manifest-to-prediction label-mapping defect was detected in the committed artifacts. Score-component semantics, classifier class ordering, candidate suppression behavior, and independent annotation quality remain unresolved.

