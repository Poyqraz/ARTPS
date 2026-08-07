# Independent eval v1 — blind validation review report

status: complete
review_type: repeat_author_blind_review
independent_annotator: false
model_blind: true
reviewed: 54/54
comparison_status: complete
decision: systematic_label_issue_detected

## Method

Model-blind repeat visual review of the 54 validation images in `independent_eval_v1`.
The reviewer saw only neutral `review_XXXX.jpg` images plus the annotation guide, choosing
`positive` / `negative` / `uncertain` / `exclude`. The reviewer did **not** see original
binary labels, terrain/category, source path, split, model scores, or candidate diagnostics.

This is a **model-blind repeat-author review**, **not** an independent second annotation.

## Completion

- reviewed: 54/54
- reviewer role: `repeat_author_review`

## Review label distribution (canonical)

- positive / 1: 46
- negative / 0: 8
- uncertain: 0
- exclude: 0

## Confidence distribution

- high: 38
- medium: 11
- low: 5

## label_review_confusion_matrix (NOT a model prediction confusion matrix)

| original \ review | review 0 | review 1 |
|---|---|---|
| original 0 | 5 | 22 |
| original 1 | 3 | 24 |

- original_positive_count: 27
- original_negative_count: 27
- reviewed_positive_count: 46
- reviewed_negative_count: 8

## Agreement / disagreement

- agreement_count: 29
- disagreement_count: 25
- agreement_rate: 0.537
- disagreement_rate: 0.463
- original_positive_to_review_negative: 3
- original_negative_to_review_positive: 22

## Decision

Predeclared thresholds (unchanged): C if `(uncertain+exclude)/n >= 0.25`; B if
binary-comparable `disagreement_rate >= 0.20`; A otherwise.

**decision: systematic_label_issue_detected (B)**

The model-blind repeat-author review identified a systematic disagreement with the original
validation labels under the predeclared 20% disagreement criterion. The disagreement is
directional: 22 of 27 originally-negative validation images were re-reviewed as positive.
This flags the validation label semantics for follow-up review; it does **not** by itself
prove the original labels are incorrect.

## Guarantees for this stage

- The `independent_eval_v1` manifest was **not** modified.
- `annotation_version` was **not** changed.
- Model scores/predictions were **not** included in the comparison.
- Freeze unchanged: `test_opened: false`, `final_test_authorized: false`,
  `FINAL_TEST_SCOPE.status: blocked_validation_sanity_review`.
- `IND_EVAL_V1` claim support unchanged (`protocol_defined_pending_data`); no manuscript
  Results numbers added.

## Provenance (committed, sanitized)

- `reproduction/iac2026/annotations/independent_eval_v1_repeat_author_blind_review.csv`
  (label-only; no path/sample_id/original label/model score)
- `reproduction/iac2026/annotations/independent_eval_v1_repeat_author_blind_review.meta.json`
  (SHA provenance for results, pack manifest, source manifest, annotation guide)

## Next action (because decision is B)

This is a **separate PR** (`data/independent-eval-v1-annotation-v1-1`); PR #28 closes only
the label-audit result.

- create `independent_eval_v1_1`
- keep the original v1 manifest immutable
- apply only completed blind-review label corrections
- preserve validation sample IDs
- audit the full 360-sample label semantics before claiming v1_1 is corrected
- re-run SHA / leakage audit
- re-run the frozen validation profile without tuning
- test remains closed

Do **not** assume the remaining train/test labels are correct just because the 54 validation
samples were re-reviewed. A systematic validation disagreement implies the label-production
method for the whole 360-sample benchmark needs auditing. Do not inspect test-split labels
using model scores, and do not open the test-split images to ARTPS.
