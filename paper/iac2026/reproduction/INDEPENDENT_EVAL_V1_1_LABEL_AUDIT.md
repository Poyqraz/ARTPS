# independent_eval_v1_1 — label audit (research + review plan)

status: research_complete_review_pending
review_type: repeat_author_blind_review
independent_annotator: false
reviewed: 54/360
pending_manual_review: 306/360
comparison_status: pending_review_completion

This document records a research-first audit of the `independent_eval_v1` label
semantics and defines the full-360 human review that produces `independent_eval_v1_1`.
It does **not** modify `independent_eval_v1`, historical manuscript results, or the
test embargo.

## Scope boundary

This audit concerns the **new supplementary benchmark** `independent_eval_v1` only.
It is not a validity audit of the historical ARTPS manuscript / accepted IAC results
(AUROC 0.894, AUPRC 0.847, F1 0.823, baseline AUROC 0.856, 28.1 FPS). Those values
are **not** invalidated, deleted, or declared wrong by this work.

## Provenance SHAs (pinned at audit time, LF-normalized)

SHAs are computed on LF-normalized content so they match on both CRLF (Windows worktree)
and LF (git blob / CI) checkouts.

| Artifact | SHA256 |
|---|---|
| `reproduction/iac2026/manifests/independent_eval_v1.csv` | `9f953dc07286738b82a07b6a4311ceaf6cc64de361893af3a04ee3279a45d408` |
| `reproduction/iac2026/annotations/independent_eval_v1_repeat_author_blind_review.csv` | `e237fdd8dcf8340c2324d00e1b3c5d3d9e93ed8ac1ed30a5bc8e5339a8dd479f` |
| `reproduction/iac2026/annotations/independent_eval_v1_repeat_author_blind_review.meta.json` | `1f0cb955c6cb704ca6ee8acce89ec4354a37687e5ab80fe20afd1fe951922153` |
| `paper/iac2026/reproduction/INDEPENDENT_EVAL_V1_ANNOTATION_GUIDE.md` | `5b4baa407a388a3f5b36b2331a4f1e718d0cd854371929e73d10a37fcb868beb` |

## Repository state at audit

- PR #28 merged to `main` (merge commit `fd5cdb5`).
- `test_opened: false`
- `final_test_authorized: false`
- `FINAL_TEST_SCOPE.status: blocked_validation_sanity_review`

## Research findings

### 1. Original 360 label-generation mechanism — VERIFIED

The original labels were **not** assigned by a human one-by-one. They were produced by
an automated classical-image heuristic in
[`scripts/iac2026/review_independent_eval_v1_visual.py`](../../../scripts/iac2026/review_independent_eval_v1_visual.py)
(`decide_label`), using only Pillow image statistics (no ML). Rule summary:

- `blob_score >= 1.75 AND std >= 18.0` -> positive (1)
- `blob_score <= 1.45` -> negative (0)
- `1.45 < blob_score < 1.75` -> uncertain
- `std < 6.0` -> excluded (featureless); plus exposure/blur/border/`/rover/` excludes

`blob_score` is an edge-energy peak/mean ratio (localized structure), i.e. an
**edge-concentration proxy**, not a science-interest judgment.

Evidence the labels were batch-generated, not interactively reviewed:
- All 1186 annotation-queue rows share one identical timestamp `2026-08-05T19:36:32Z`.
- `annotation_notes` carry machine metrics (`blob=`, `featureless_std=`, `qc:`).
- `annotator_id` = `workspace_visual_review` for all rows (a script identity, not a person).

### 2. Role of Roboflow class / folder names — VERIFIED

Folder/terrain names (`boulder`, `rocky`, `hills_or_ridge`, `flat_terrain`, `dusty`,
`rover`, `Unlabeled`) were used only for **domain inclusion** (Curiosity-Mastcam path
filter) and to exclude `/rover/`. They were **not** mapped to `binary_label`. The domain
doc states explicitly: "Terrain folder names are not `binary_label` ground truth."

### 3. How 180/180 balance arose — VERIFIED (forced)

`_balance_included(target_included=360)` in the same script deterministically caps the
included set to ~50/50 by re-marking majority-class excess as `excluded`
(note `;balance_or_cap_exclude`). Result: exactly 180/180 overall and per split
(train 126/126, validation 27/27, test 27/27). The balance was **forced**, not natural.

### 4. Annotation guide vs. historical ARTPS target semantics — PARTIALLY VERIFIED

The current annotation guide defines positive as a human-judged science-interest target
a rover operator would flag. The historical manuscript concerns anomaly/novelty detection
performance on a separate (unrecovered) dataset. The v1 guide intentionally does **not**
claim equivalence to historical C05/C06 ground truth. The heuristic labels do **not**
implement the guide's human science-interest definition.

### 5. Likely semantic cause of the 25/54 disagreement — VERIFIED

In PR #28's model-blind repeat-author review of the 54 validation images:
original 27/27 vs. review 46/8; agreement 29, disagreement 25 (rate 0.463); direction
22 of 27 originally-negative -> review-positive. This is consistent with the heuristic
labelling low-edge-concentration frames as negative while a guide-aligned human marks
many of them as science-interest positive. The disagreement reflects a
label-semantics mismatch in the new benchmark, not evidence about the historical dataset.

### 6. Label provenance for the 306 non-validation samples — VERIFIED

Of 360 samples, only the 54 validation images carry genuine human (repeat-author)
review (PR #28). The remaining **306 (252 train + 54 test)** carry only the automated
heuristic provenance. Their true human labels are **UNKNOWN** -> `pending_manual_review`.

### 7. Is v1_1 evidence-justified? — VERIFIED (yes)

The label-production rule does not match the intended benchmark definition (automated
edge heuristic + forced balance, recorded under a misleading `workspace_visual_review`
provenance). Justification comes from the label-production evidence and the model-blind
audit, not from ARTPS metric weakness. Therefore a new annotation version
`independent_eval_v1_1` built from genuine full human review is warranted. The original
`independent_eval_v1` remains immutable.

## Plan (agent assigns no labels)

1. Build a neutral review pack for the 306 non-validation samples with continuation IDs
   `review_0055`..`review_0360` (the 54 validation reviews from PR #28 stay immutable).
2. Author reviews all 306 model-blind in Streamlit (`--repeat-author-review`).
3. Fail-closed gate: no `independent_eval_v1_1` manifest and no validation rerun until
   360/360 genuine human reviews exist.
4. After 360/360: full original-vs-review comparison (label-only, no model scores),
   then freeze `independent_eval_v1_1` with per-row provenance; real class distribution
   (no forced balance).
5. Re-run the frozen `artps_full_frozen_mars_clf_on_v1` profile (FP32) on v1_1 validation
   only; test stays embargoed.

## Invariants

- `independent_eval_v1` stays byte-for-byte immutable.
- Historical manuscript / accepted-IAC values unchanged.
- C05/C06/C07 stay `accepted_abstract_reproduction_pending` (means historical artifact
  reproduction pending, **not** invalid).
- `test_opened: false`, `final_test_authorized: false`.
- No model-score-driven or folder-derived labels; no fabricated human review.
