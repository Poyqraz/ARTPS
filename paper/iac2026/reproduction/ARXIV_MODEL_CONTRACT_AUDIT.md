# arXiv model contract audit (source-derived vs code-derived)

Audit scope: model/checkpoint contracts referenced by ARTPS and frozen independent_eval_v1 runners.
Source archive: `docs/arxiv_clean.zip` (paper + figures; **not** copied into experiment result bundles).

## Labeling convention

| Class | Meaning | Examples in this repo |
|-------|---------|------------------------|
| **source-derived** | Stated in arXiv/paper text or tables; not re-measured here | Accepted abstract C05–C07 headline numbers; paper architecture names |
| **code-derived** | Implemented and verified in repository code/tests | `FrozenARTPSConfig` defaults, registry SHA-256, protocol lock fields |
| **unverified legacy** | Local artifact present but training provenance unknown | `frozen_checkpoint_registry.yaml` entries with `training_dataset_provenance: unverified` |

## Component audit

| Component | Paper / arXiv claim (source-derived) | Code contract (code-derived) | Audit status |
|-----------|--------------------------------------|------------------------------|--------------|
| Autoencoder | Curiosity-oriented AE reconstruction | `OptimizedAutoencoder`, 128px, latent 1024, `results/optimized_autoencoder_curiosity_extended.pth` SHA pinned | SHA + strict load in registry verifier |
| DPT depth | Relative depth for fusion | `MiDaSDepthEstimator` `DPT_Large`, `local_state_dict` only for primary eval | Fail-closed if fallback/hub |
| Depth classifier | 5-class known-value signal | `DepthEnhancedClassifier`, optional via profile YAML | Disabled profiles documented |
| PaDiM / PatchCore | Historical baseline mentions | Embedded stats/bank in legacy `.pth`; exploratory runner only | Not primary baseline; no 0.856 reproduction |
| Curiosity score | α·known + β·anomaly | `src/core/curiosity_scorer.py` + detection core scoring | Independent eval uses image-level max candidate score |

## Rules for experiment bundles

1. **Do not** copy paper numeric results into `results/iac2026/independent_eval_v1/` artifacts.
2. Mark validation outputs `not_final_test_result: true` until test embargo opens.
3. Distinguish `evaluation_role: secondary_exploratory` for legacy baselines.
4. Prefer registry SHA verification over path existence checks alone.

## Gaps requiring author attestation (not closed by this audit)

- Exact training datasets for legacy `.pth` checkpoints.
- Whether paper-reported detection metrics used identical preprocessing (`raw_rgb_v1` vs `mars_enhancement_v1`).
- Hardware-specific timing numbers (C07 track separate).

## Related files

- `reproduction/iac2026/frozen_checkpoint_registry.yaml`
- `src/artps_inference.py` (`FrozenARTPSConfig`, `load_frozen_artps_profile`)
- `paper/iac2026/reproduction/FROZEN_CHECKPOINT_EVALUATION_LIMITATIONS.md`
