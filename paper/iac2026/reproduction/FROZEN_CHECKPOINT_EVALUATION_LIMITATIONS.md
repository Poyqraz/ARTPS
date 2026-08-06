# Frozen checkpoint evaluation limitations (independent_eval_v1)

Designation for this evaluation track:

**frozen-checkpoint current evaluation with unverified training provenance**

This document states what frozen ARTPS validation **can** and **cannot** establish.

## What is frozen

- Checkpoint bytes are pinned in `reproduction/iac2026/frozen_checkpoint_registry.yaml` (SHA-256, size).
- Four predeclared ARTPS full-profile YAMLs under `reproduction/iac2026/configs/independent_eval_*artps*`.
- Protocol lock SHA: `7767f695746d0237803f57ffd2fef8f96a1434fca5d2f2ffaf2c799c3187dfe9`.
- Test split remains embargoed until `reproduction/iac2026/test_freeze/TEST_OPEN_STATUS.yaml` sets `test_opened: true`.

## What validation runs establish

- **Validation-only** profile comparison on `independent_eval_v1` (AUROC, AP, F1 threshold on validation).
- Reproducible inference bundles under `results/iac2026/independent_eval_v1/validation/` with provenance sidecars.
- Optional FP32 vs AMP parity checks (`check_precision_parity.py`) with fixed gates (`1e-4`).
- Measurement of the **current frozen deployed system** on the pinned Curiosity Mastcam Roboflow benchmark.

## What is **not** established

| Limitation | Reason |
|------------|--------|
| Historical C05/C06 numbers (0.894 / 0.847 / 0.823 / 0.856) | Different protocol, dataset, and training provenance; not a reproduction of historical claims |
| Training dataset provenance for legacy `.pth` artifacts | Registry marks `training_dataset_provenance: unverified`; exact historical sample IDs unavailable |
| Leakage-free / independent training / unseen / external test | Overlap between checkpoint training data and current benchmark **cannot be conclusively excluded** |
| Clean retraining-based generalization experiment | No retraining occurred; checkpoints predate `independent_eval_v1` |
| Metric distance claims from arXiv paper tables | Paper numbers are source-derived; not copied into experiment bundles |
| Primary manuscript results before test embargo opens | All validation artifacts carry `not_final_test_result: true` |
| Legacy PaDiM/PatchCore as primary baselines | `evaluation_role: secondary_exploratory`; not associated with historical 0.856 |

Canonical anomaly implementations for this repository live under `src/models/anomaly/`. The `ARTPS/src/models/anomaly/` tree is a legacy mirror kept byte-aligned for historical imports; this PR does not refactor that duplication.

Do **not** describe this evaluation as leakage-free, independent training, unseen data, or external test unless later evidence establishes those statements.

## Operational constraints

- DPT is **relative depth only** (within-image ordering); not metric distance.
- Classifier-off profiles use `known_value=0.5` fallback (documented in inference warnings).
- Cache arrays (`*.npz`, `*.npy`) under `cache/` are gitignored; commit small CSV/JSON validation outputs only.
- GPU + local weights required for real runs; CI uses synthetic unit tests only.
- Annotation quality note: primary benchmark used single-reviewer workspace visual review (`independent_double_review: false`).

## Runner entrypoints

| Script | Purpose |
|--------|---------|
| `scripts/iac2026/run_artps_frozen_full_profile.py` | Batch inference + prediction CSV/JSONL |
| `scripts/iac2026/generate_artps_full_profile_cache.py` | Train/validation cache generation |
| `scripts/iac2026/select_frozen_validation_profile.py` | Validation-only profile selection |
| `scripts/iac2026/check_precision_parity.py` | FP32 vs AMP parity gate |
| `scripts/iac2026/run_legacy_baseline_exploratory.py` | PaDiM/PatchCore exploratory only |
