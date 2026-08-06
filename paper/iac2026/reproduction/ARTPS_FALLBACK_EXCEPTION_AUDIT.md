# ARTPS Exception / Fallback Audit (independent_eval_v1)

Audit of silent zeros and fallbacks in the frozen full-profile inference path.
Does **not** open the test split or change `profile_selection.json`.

## Findings

### 1. Broad `except Exception` → `image_score=0.0`

`src/artps_inference.predict_image` catches all exceptions, sets:

- `processing_status = "error"`
- `image_score = 0.0`
- `warning_flags` includes `str(exc)`
- `anomaly_mse = None`

This can pollute ranking if errors are frequent (zeros treated as low anomaly under higher-is-anomalous).
**Objective-bug rule:** making this fail-loud is a correctness fix, not tuning — requires `…_v1_1` config id, full validation re-run, and `profile_selection_v1_1.json` (old selection immutable). This PR documents only; no silent-fallback patch unless a failing primary-path test proves mass errors.

### 2. Classifier-off → `known_value=0.5`

When `enable_classifier` is false or classifier missing, known value is fixed at `0.5` with warning
`classifier_disabled_known_value_fallback_0.5`. Explicit, reproducible fallback.

### 3. DPT `None` → `RuntimeError`

`_depth_for_fusion` raises if depth is `None`. No soft depth fallback in primary frozen path — preferred for eval integrity.

### 4. No candidates after masks → `0.0` with `status=ok`

Not an exception. Aggregation contract: max over empty set is defined as `0.0`.
Committed primary JSONL shows this pattern with `anomaly_mse > 0` on many rows.

## Primary validation JSONL check

Audit script classifies zero scores via committed JSONL only (no GPU re-inference).
If `processing_status=error` rows appear on the selected profile, “full frozen” primary status is rejected until fail-loud / `v1_1` remediation.

## Decision (this PR)

- Default: no objective silent-fallback mass-error proven from committed artifacts → **no auto `v1_1`**.
- Final test stays `blocked_validation_sanity_review` / `final_test_authorized: false`.
