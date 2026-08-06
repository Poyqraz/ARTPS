# ARTPS Score Semantics Audit (independent_eval_v1)

Code-reading audit of how the frozen full-profile pipeline builds `image_score`.
This document does **not** authorize score flips, retunes, or final-test opening.

## Contract (as implemented)

| Claim | Value | Verified from |
|---|---|---|
| Positive label | `1` | `run_artps_frozen_full_profile.py` (`y_true = int(row["binary_label"])`) |
| Higher score means more anomalous | `true` (intended) | `detection_metrics_lib.orient_scores` + runner metrics config |
| Decision rule | `score >= threshold` | `select_threshold_on_validation` / confusion at selected `t` |
| Image aggregation | `max` over kept candidate scores; **no candidates → 0.0** | `src/artps_inference.py` `predict_image` |

## Component table

| Component | Variable | Range (typical) | Higher means | Transform into final image score? | Used in final image score? | Verified from |
|---|---|---|---|---|---|---|
| AE reconstruction error | `anomaly_mse` / MSE map | ≥0 | more reconstruction residual | Feeds combined anomaly map via recon weight; **not** the image score itself | Indirect | `artps_inference._ae_forward`, `compute_combined_anomaly_map` |
| Combined anomaly map | `combined_map` pools | [0,1] after normalize | more anomalous texture/recon/depth mix | Pooled into `combined_pool` then fusion | Yes (via fusion) | `artps_detection_core.compute_combined_anomaly_map` |
| Depth map | relative DPT depth | image-relative | **farther** in UI convention used by maps | Proximity uses `1.0 - depth` | Indirect | `artps_inference._depth_for_fusion`; `proximity_w = normalize(1-depth)` |
| Depth pool on box | `depth_pool` | [0,1] | deeper / farther pool value | `anomaly_score` term `0.05 * (1 - depth_pool)` (near gets small bump) | Yes | `_fuse_object_scores` |
| Classifier known value | `known_value` = `argmax/4` | {0,0.25,0.5,0.75,1} | higher class index / “known” | `local_value` 0.55 weight; gates final via `anomaly * (0.70 + 0.30*local)` | Yes when classifier on | `_known_value_score`, `_fuse_object_scores` |
| Classifier-off fallback | `known_value = 0.5` | fixed | neutral | Same fusion path; warning `classifier_disabled_known_value_fallback_0.5` | Yes | `predict_image` |
| Detector confidence | `detector_conf` | [0,1] | stronger detection support | `0.10 * detector_conf` in anomaly_score | Yes | `_fuse_object_scores` |
| PaDiM / PatchCore pools | `padim_pool` / `patchcore_pool` | [0,1] | more anomalous | Weighted in anomaly_score; **frozen path passes `None` → 0 pools** | Yes (zero in frozen full) | `predict_image` (`padim_map=None`, `patchcore_map=None`) |
| Object final score | `det["score"]` | [0,1] clipped | more anomalous candidate | Image score = max over kept | Yes | `_fuse_object_scores` → `final_score` |
| FP / policy masks | keep/drop detections | boolean | N/A | Dropped boxes never enter max | Yes (by exclusion) | `_should_keep_detection` / scoring filters |
| Exception fallback | any `Exception` in `predict_image` | N/A | N/A | Sets `image_score=0.0`, `processing_status=error` | Degenerate zero | `predict_image` broad `except Exception` |
| Empty candidate set | `scored == []` | N/A | N/A | `image_score=0.0` with `processing_status=ok` | Degenerate zero | `predict_image` |

## Fusion formula (object level)

From `_fuse_object_scores`:

- `local_value = clip(0.55*known + 0.25*depth_pool + 0.20*combined_pool)`
- `anomaly_score = clip(0.50*combined + 0.20*padim + 0.15*patchcore + 0.10*detector_conf + 0.05*(1-depth_pool))`
- `final_score = clip(anomaly_score * (0.70 + 0.30*local_value))`

Image score = **max** `final_score` among kept candidates.

## Fail-loud vs silent paths

| Path | Behavior | Audit note |
|---|---|---|
| DPT returns `None` | `RuntimeError` (no depth fallback) | Fail-loud — good for primary eval |
| Broad `except Exception` in `predict_image` | `image_score=0.0`, status=`error` | Silent-ish zero; fail-loud would be a separate fix + `…_v1_1` revalidation |
| Classifier disabled | known=0.5 + warning | Explicit fallback, not an exception |
| No valid candidates after masks | score=0.0, status=`ok` | Contractual aggregation, not an exception |

Primary validation JSONL for `artps_full_frozen_mars_clf_on_v1` shows many `image_score=0` / `candidate_count=0` / `processing_status=ok` rows — consistent with **no-candidate → 0**, not mass exception fallback.

## Orientation diagnostic policy

Negated / one-minus AUROC+AP may be computed for diagnosis only.
They **must not** be promoted or used to rewrite `profile_selection.json` unless an objective production score-orientation bug is proven.

## Outcome for this PR

- Semantics traced; no automatic orientation flip.
- Degenerate threshold 0.0 + below-chance ranking remain blockers.
- No new profile / retune in this audit.
