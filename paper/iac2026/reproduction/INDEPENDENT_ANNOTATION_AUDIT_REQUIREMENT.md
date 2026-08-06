# Independent annotation audit requirement (independent_eval_v1)

Validation labels remain historically fixed for the frozen selection artifact.
This PR does **not** change any `binary_label`.

## Requirement

An independent blind review of the 54 validation images is required before any
consideration of label quality as an explanation for below-chance ranking.

Queue (labels/scores/split hidden):

`reproduction/iac2026/annotations/independent_eval_v1_validation_blind_review_queue.csv`

- Deterministic shuffle seed: `20260806`
- `audit_status: pending_independent_review`
- Do not use model scores during review
- Do not open the test split while review is pending

## Policy

Score-based label edits are forbidden. Any future label change needs a separate
adjudication protocol and would invalidate prior selection only via a new
versioned selection artifact — never by mutating `profile_selection.json`.
