# Post-merge diagnosis findings (independent_eval_v1)

Date context: after PR #26 merge (`7648703`). Final test **not** opened.

## Freeze (unchanged)

| Field | Value |
|---|---|
| `TEST_OPEN_STATUS.test_opened` | `false` |
| `FINAL_TEST_SCOPE.status` | `blocked_validation_sanity_review` |
| `final_test_authorized` | `false` |

No `experiments/independent-eval-v1-final-test` branch. No test predictions.

## What ran

1. Blind-review pack built under gitignored `results/iac2026/independent_eval_v1/blind_review_pack/` (54 images, SHA verified). Operator notes: [`BLIND_REVIEW_OPERATOR_NOTES.md`](BLIND_REVIEW_OPERATOR_NOTES.md). Independent review still **pending**; author-repeat meta only.
2. Instrumented validation-only rerun on frozen `artps_full_frozen_mars_clf_on_v1` with `ARTPS_DATASET_ROOT=mars_images/`.
3. Output: [`component_diagnostics_v1.csv`](../../../results/iac2026/independent_eval_v1/validation/artps_full_frozen_mars_clf_on_v1/component_diagnostics_v1.csv) — `execution_path=instrumented_validation_rerun`.

## Score / component bug?

- **image_score parity vs committed predictions.csv: 54/54 OK** (SHA unchanged `24df219b…`).
- No `processing_status=error`, no classifier/DPT/AE fallback warnings on this run.
- Therefore: **no objective score-path bug proven** that would authorize `…_v1_1` from this diagnosis alone.

## Zero-score / suppression (selected profile)

Five validation images remain at `image_score=0` with **raw_proposal_count > 0** (not “no proposal”):

| `no_valid_candidate_reason` | count |
|---|---:|
| `field_scale_rejection` | 3 |
| `size_distance_policy_rejection` | 1 |
| `candidate_score_filtering` | 1 |

Fine reasons are now filled (no longer `unavailable_requires_instrumented_validation_rerun` for these rows). Changing masks/thresholds to “fix” zeros would be **tuning**, out of scope here.

## Blind annotation

Independent second-annotator review is **not** complete (`independent_review_status: pending`).  
No `annotation_version` bump and no label edits from this phase.

## Decision gate (this phase)

| Hypothesis | Verdict |
|---|---|
| Label quality issue | **Unresolved** — pending independent blind review |
| Objective score/component bug | **Not found** on parity + frozen path |
| System non-discriminative on this benchmark | **Still the working acceptance** for ranking (below-chance AUROC/AP + degenerate threshold=0 remain) |

**Next (separate work, still test-closed):** complete independent blind review of 54 validation images; only then consider annotation_version bump / revalidation, or accept non-discriminative and keep manuscript claims unre-measured.

**Do not:** open test split, retune fusion/threshold, promote orientation flip, or auto-create `v1_1`.
