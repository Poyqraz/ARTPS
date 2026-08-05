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
| Image score | max-pool of the anomaly map over spatial locations |
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

- **Unit:** one Mars surface RGB image (`sample_unit=image`).
- **Label:** binary. `1` = anomalous (human-verified presence of at least one anomalous region or object of scientific interest under the labeling guide for this version). `0` = not anomalous under that guide.
- **Semantics field:** `label_semantics=anomaly_binary`.
- **Provenance fields (required on every row):** `label_source`, `annotation_version=independent_eval_v1`.
- **Not claimed here:** equivalence to any unpublished historical GT used for the accepted abstract table.

Until a labeled, SHA-pinned set exists, the manifest remains header-only:
[`reproduction/iac2026/manifests/independent_eval_v1.template.csv`](../../../reproduction/iac2026/manifests/independent_eval_v1.template.csv).

---

## Group-aware split rules

1. Assign every sample to exactly one of `train` / `validation` / `test`.
2. Populate non-empty `scene_group_id` and `duplicate_group_id` for every row.
3. Fail the audit if any `sha256`, `scene_group_id`, or `duplicate_group_id` appears in more than one split.
4. Validation and test must each contain both classes `{0,1}`.
5. Random seed for any future split generation must be recorded in the run config (`random_seed`).

---

## Metrics and threshold recipe

1. Fit / bank models on `train` only (document `train_bank_recipe`).
2. Score `validation` and `test` with the same frozen model.
3. Select threshold on validation by maximizing F1; ties → highest threshold.
4. Report on test: AUROC, average precision (AUPRC), F1 at the frozen threshold.
5. Do not bootstrap (`bootstrap_iterations=0` until implemented).

---

## Baseline recipe (PaDiM / PatchCore)

- Run **PaDiM** and **PatchCore** as independent configs / prediction tables.
- Do **not** invent a combined “PaDiM/PatchCore (WRN-50-2)” cell that averages scores to imitate 0.856.
- Stubs remain fail-loud until `weights_sha256` and `train_bank_recipe` are real
  ([`scripts/iac2026/baselines/base.py`](../../../scripts/iac2026/baselines/base.py)).
- No anomalib dependency in this harness.

---

## Reporting rules

- Section / table title: **Current reproducible evaluation (`independent_eval_v1`)**.
- Always cite that accepted abstract C05/C06 figures are separate historical claims.
- Bundle outputs under `results/iac2026/reproduction/<run_id>/` with
  `evaluation_purpose=current_reproducible_evaluation` and `claim_ids=["IND_EVAL_V1"]`.

---

## Next-step gate (out of scope for this pin)

No real_evidence run under this protocol until all of:

1. Non-template manifest rows with real 64-hex `sha256` (no `SYNTHETIC_*`)
2. Labels under `annotation_version=independent_eval_v1`
3. Images reachable via `dataset_root_env` and hash-matched
4. Baseline / ARTPS checkpoint SHA + train bank recipe pinned
5. Fresh passing `audit_reproduction_inputs.py` for the run config

Historical C05/C06 readiness
(`check_c05_c06_definition_readiness.py`) stays blocked until historical artifacts
are recovered; this protocol does not unlock that gate.
