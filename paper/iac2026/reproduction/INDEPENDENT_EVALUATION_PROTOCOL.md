# Independent evaluation protocol v1

**Purpose:** `current_reproducible_evaluation`

**Protocol id:** `independent_eval_v1`

**Claim id for outputs:** `IND_EVAL_V1`

This protocol is **not** a reproduction of accepted-abstract C05/C06 numbers
(0.894 AUROC, 0.847 AUPRC, 0.823 F1, 0.856 baseline AUROC). Those remain
`accepted_abstract_reproduction_pending` because historical split / prediction /
checkpoint / notebook artifacts were not recovered
([C05_C06_RESPONSE_STATUS.md](C05_C06_RESPONSE_STATUS.md),
[DATASET_MANIFEST_GAPS.md](DATASET_MANIFEST_GAPS.md)).

Machine lock: [`reproduction/iac2026/INDEPENDENT_EVAL_V1.yaml`](../../../reproduction/iac2026/INDEPENDENT_EVAL_V1.yaml).

---

## Explicit non-equivalence

| Forbidden | Required |
|-----------|----------|
| “reproduced C05/C06” | Report as **current reproducible evaluation** |
| “matches abstract 0.894 / 0.847 / 0.823 / 0.856” | State accepted numbers are historical claims with unrecovered artifacts |
| Averaging PaDiM + PatchCore into a fake 0.856 | Separate baseline runs only |
| Inventing per-file rows from 2847 / 1247 / 892 / 708 aggregates | SHA-pinned labeled manifest only |

Ledger support for C05/C06 must **not** move to `measured` from this protocol alone.

---

## Locked contract

| Field | Value |
|-------|--------|
| `task_level` | `image_binary` |
| Positive class | `binary_label=1` = image-level anomaly per annotation guide below |
| Score orientation | higher = more anomalous |
| Image score | `max_valid_candidate_after_masks` (max among localization candidates after rover/border/telemetry masks; no candidate → `0.0`). Raw single-pixel `max_pool_anomaly_map` is **not** primary. |
| Splits | `train` / `validation` / `test` |
| Group-aware leakage | same `scene_group_id`, `duplicate_group_id`, or file `sha256` must not appear in more than one split (**error**) |
| PR metric | `average_precision` (sklearn average precision) |
| AUROC | ROC-AUC on image scores vs `y_true` |
| Threshold policy | `validation_selected` |
| Selection metric | `f1` on validation |
| Tie-break | `highest_threshold` |
| Test F1 | uses the validation-selected threshold only (no re-tuning on test) |
| Baselines | PaDiM and PatchCore as **separate** runs; required keys: backbone, layers, image_size, weights_path, weights_sha256, score_aggregation, train_bank_recipe (+ coreset_ratio for PatchCore) |
| `evaluation_purpose` | `current_reproducible_evaluation` |
| `claim_ids` | `["IND_EVAL_V1"]` only |

---

## Annotation definition (`annotation_version: independent_eval_v1`)

Full rules: [INDEPENDENT_EVAL_V1_ANNOTATION_GUIDE.md](INDEPENDENT_EVAL_V1_ANNOTATION_GUIDE.md).
Dataset acquisition: [INDEPENDENT_EVAL_V1_DATASET_PLAN.md](INDEPENDENT_EVAL_V1_DATASET_PLAN.md).

- **Unit:** one Mars surface RGB image (`sample_unit=image`).
- **Label:** binary. `1` = anomalous (human-verified presence of at least one anomalous region or object of scientific interest under the labeling guide for this version). `0` = not anomalous under that guide.
- **Semantics field:** `label_semantics=anomaly_binary`.
- **Provenance fields (required on every row):** `label_source`, `annotation_version=independent_eval_v1`.
- **Not claimed here:** equivalence to any unpublished historical GT used for the accepted abstract table.

Until a labeled, SHA-pinned set exists, the manifest remains header-only:
[`reproduction/iac2026/manifests/independent_eval_v1.template.csv`](../../../reproduction/iac2026/manifests/independent_eval_v1.template.csv).

---

## Image score aggregation (primary)

Primary method: **`max_valid_candidate_after_masks`**.

- Anomaly map stage: post-localization candidates
- Apply rover / border / telemetry masks before scoring
- Valid area: candidates that survive masks only
- Multiple candidates: take the maximum candidate score
- No valid candidate remains: image score `0.0`
- Raw single-pixel `max_pool_anomaly_map` is forbidden as the primary metric (noise-pixel risk)
- Sensitivity analyses may be documented later but must not replace the primary method without a new protocol version

---

## Group-aware split rules

1. Assign every sample to exactly one of `train` / `validation` / `test`.
2. Populate non-empty `scene_group_id` and `duplicate_group_id` for every row.
3. Fail the audit if any `sha256`, `scene_group_id`, or `duplicate_group_id` appears in more than one split.
4. Validation and test must each contain both classes `{0,1}`.
5. Random seed for any future split generation must be recorded in the run config (`random_seed`).
6. `split_ratios` stay `PENDING_RATIO_SELECTION` until labeled volume is known; builder refuses historical aggregate quotas.
7. After test split creation, freeze is immutable; mutating test requires a new protocol/run version
   (`scripts/iac2026/build_independent_eval_split.py`).

---

## Metrics and threshold recipe

1. Fit / bank models on `train` only (document `train_bank_recipe`).
2. Score `validation` and `test` with the same frozen model.
3. Select threshold on validation by maximizing F1; ties → highest threshold.
4. Report on test: AUROC, average precision (AUPRC), F1 at the frozen threshold.
5. Do not bootstrap (`bootstrap_iterations=0` until implemented).
6. Bootstrap CI / paired comparisons are a **blocker before manuscript results** (see lock YAML `statistical_reporting_plan`).

---

## Baseline recipe (PaDiM / PatchCore)

- Train bank uses **only** `binary_label=0` (normal) images.
- Validation and test contain both classes; same manifest and same test IDs across ARTPS/baselines.
- Run **PaDiM** and **PatchCore** as independent configs / prediction tables / run IDs.
- Do **not** invent a combined “PaDiM/PatchCore (WRN-50-2)” cell that averages scores to imitate 0.856.
- Stubs remain fail-loud until `weights_sha256` and `train_bank_recipe` are real
  ([`scripts/iac2026/baselines/base.py`](../../../scripts/iac2026/baselines/base.py)).
- No anomalib dependency in this harness.

### ARTPS selection

- Separate config / run ID; pin checkpoint SHA, modules enabled, learned-depth on/off, AE on/off,
  score generation path, and `model_selection_policy=validation_only_no_test_peek`.
- Do not choose model profiles using test-set outcomes.

### Test embargo

- Preprocessing / hyperparameters / threshold: validation only.
- Test split opens only for the final frozen run.
- Config change after test peek → new protocol or run version.
- Failed runs must be registered; reporting only the best run is forbidden.
- Planned model/config IDs are pre-listed before evaluation.

---

## Protocol versioning

- `protocol_version: 1.0.0`, `protocol_status: draft_pending_data`
- `created_before_dataset_labeling: true`, `created_before_model_evaluation: true`
- Typo-only → patch; metric/threshold/annotation change → minor/major; changing primary
  metric after seeing data → **new protocol id**
- After first real manifest rows exist, treat protocol lock SHA as immutable for that dataset generation
- Every run bundle must record `protocol_lock_sha256`

Runtime enforcement: `scripts/iac2026/independent_eval_contract.py` (loaded by config validation).

---

## Reporting rules

- Section / table title: **Current reproducible evaluation (`independent_eval_v1`)**.
- Always cite that accepted abstract C05/C06 figures are separate historical claims.
- Bundle outputs under `results/iac2026/reproduction/<run_id>/` with
  `evaluation_purpose=current_reproducible_evaluation` and `claim_ids=["IND_EVAL_V1"]`.
- Metrics must set `historical_claim_reproduction=false` and `eligible_for_C05_C06_closure=false`.
- SW outputs: `eligible_for_IND_EVAL_V1_result_reporting=false`.
- Real outputs remain `candidate_real_evidence` with `author_verified=false` until registry pin.

---

## Next-step gate (out of scope for this pin)

No real_evidence run under this protocol until all of:

1. Non-template manifest rows with real 64-hex `sha256` (no `SYNTHETIC_*`)
2. Labels under `annotation_version=independent_eval_v1`
3. Images reachable via `dataset_root_env` and hash-matched
4. Baseline / ARTPS checkpoint SHA + train bank recipe pinned
5. Fresh passing `audit_reproduction_inputs.py` for the run config
6. `split_ratios` unlocked with a protocol version bump (not historical aggregates)

Historical C05/C06 readiness
(`check_c05_c06_definition_readiness.py`) stays blocked until historical artifacts
are recovered; this protocol does not unlock that gate.
