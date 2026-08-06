# Depth-Enhanced Classifier Semantics Audit (independent_eval_v1)

Status: `PARTIAL_TRAINING_SCRIPT_ONLY` / checkpoint linkage `UNKNOWN_EVIDENCE_NOT_LOCATED`

`classifier_class_semantics_verified: false`

This audit documents what the training script and inference code claim. It does **not**
authorize changing `known_value = argmax / 4` in this PR.

## Questions

### 1. How many classes?

**5** (`num_classes=5`, indices 0–4).

Verified from: [`src/models/depth_enhanced_classifier.py`](../../../src/models/depth_enhanced_classifier.py)
(`DepthEnhancedClassifier`, training `main`), [`src/artps_inference.py`](../../../src/artps_inference.py)
(load path).

### 2. What does each index represent?

Training dataset builder maps **terrain folder names** to integer `value_label`:

| Index | Training name (TR) | Terrain folder keys |
|---:|---|---|
| 0 | Değersiz | `rover` |
| 1 | Düşük | `flat_terrain`, `dusty` |
| 2 | Orta | *(unused in folder map)* |
| 3 | Orta-Yüksek | `boulder`, `hills_or_ridge` |
| 4 | Yüksek | `rocky` |

Source: `MarsValueDataset.value_labels` / `value_names` in `depth_enhanced_classifier.py`.

### 3. Ordinal or categorical only?

**Intended ordinal in the training script** (science-value ranking 0→4). CrossEntropy training
still treats classes as discrete categories; the ordinal assumption is injected later by
`argmax / 4.0` at inference.

### 4. Is `argmax / 4` a defined ordinal normalization?

It is a **linear map from class index to [0, 1]** used in
[`_known_value_score`](../../../src/artps_inference.py). It is scientifically justified **only if**
(a) class indices are ordinal science-interest ranks and (b) the loaded checkpoint was trained
with that same index↔meaning map. (a) is stated in the training script; (b) is not proven for
the frozen registry SHA.

### 5. Does higher class index mean more science-interest / more anomalous?

Under the **training script’s** folder→label map: higher index ≈ higher assigned science value
(rocky=4 … rover=0). That is **not** the same as the independent_eval binary anomaly label, and
must not be assumed without checkpoint provenance.

### 6. Was the frozen checkpoint trained with this exact mapping?

**UNKNOWN_EVIDENCE_NOT_LOCATED** for the pinned classifier SHA in the frozen registry.
No checkpoint metadata sidecar with class-name list / label map hash was located. Prior
frozen-eval docs already mark training provenance as unverified.

## Decision for this PR

- Do **not** flip or retune `known_value = argmax / 4`.
- Keep `classifier_class_semantics_verified: false` in the sanity report.
- Final test remains blocked partly on `classifier_class_semantics_unverified`.
