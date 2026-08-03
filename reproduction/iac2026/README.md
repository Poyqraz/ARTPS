# IAC 2026 reproduction harness

Evidence software for claims **C05–C07**. Manuscript: `paper/iac2026/`. Accepted-abstract numbers are never pass/fail targets. Synthetic runs are **software verification only**.

## When `measured` is allowed

Only after archaeology closes task_level, labels, score definition, split, threshold policy, baseline identity (C06), and a real run is registered in `evidence_registry.json` with durable URI+SHA. Until then ledger stays `accepted_abstract_reproduction_pending`.

## Commands

### Software-verification audit

```text
python scripts/iac2026/audit_reproduction_inputs.py --config reproduction/iac2026/configs/detection_reproduction.synthetic.yaml --software-verification --run-id sw_audit
```

### Real-evidence audit (requires dataset root env + pinned files)

```text
set ARTPS_DATASET_ROOT=C:\path\to\dataset_root
python scripts/iac2026/audit_reproduction_inputs.py --config path\to\real_detection.yaml --run-id real_audit
```

### Software-verification metrics

```text
python scripts/iac2026/reproduce_detection_metrics.py --config reproduction/iac2026/configs/detection_reproduction.synthetic.yaml --software-verification --run-id sw_metrics
```

### Real metrics with passing audit

```text
python scripts/iac2026/audit_reproduction_inputs.py --config path\to\real_detection.yaml --run-id real_audit
python scripts/iac2026/reproduce_detection_metrics.py --config path\to\real_detection.yaml --audit-json results/iac2026/reproduction/real_audit/input_audit.json --run-id real_metrics
```

### Real metrics: image_binary only

The scalar metrics runner accepts **`image_binary` only**. `pixel_binary` / `region_binary` are rejected with an explicit contract error (dedicated spatial/region prediction support is out of scope). Prior `--audit-json` is compare-only; a fresh `audit_inputs` always runs and must pass.

### Historical exact C07 (manifest-pinned real inputs)

Real C07 requires `input_manifest` + `dataset_root_env` (SHA/path/order pinned). `images_dir`-only is rejected.

```text
set ARTPS_DATASET_ROOT=C:\path\to\dataset_root
python scripts/benchmark_cv_core_speed.py --config reproduction/iac2026/configs/c07_historical_exact.example.yaml --run-id c07_hist
```

### Current enhancement surrogate (supplementary; not accepted C07 claim)

Profile `current_enhancement_historical_surrogate` times current enhancement + historical recon/depth/fusion. It does **not** close the accepted-abstract 28.1 FPS claim.

```text
python scripts/benchmark_cv_core_speed.py --config reproduction/iac2026/configs/c07_current_production.example.yaml --run-id c07_surrogate
```

### Software-verification C07

```text
python scripts/benchmark_cv_core_speed.py --config reproduction/iac2026/configs/c07_software_verification.example.yaml --software-verification --run-id c07_sw
```

### C07 timed stage scope

Headline latency is `process_frame_*` total: **resize + enhance + recon surrogate + fallback depth + `fusion_localization_combined`**. Disk enumeration is outside timed scope. Decode is outside unless measured as `frame_fetch` (not in the headline total). There is **no** fabricated 70/30 split of fusion vs localization — one measured stage only.

### Output schema validation

```text
python -c "import json,jsonschema; from pathlib import Path; m=json.loads(Path('results/iac2026/reproduction/sw_metrics/detection_metrics.json').read_text()); s=json.loads(Path('reproduction/iac2026/schemas/detection_metrics.schema.json').read_text()); jsonschema.Draft202012Validator(s).validate(m); print('ok')"
```

### Evidence registry update

After uploading a real run archive to GitHub Releases, Zenodo, or another durable store, append an entry to `reproduction/iac2026/evidence_registry.json` matching `evidence_registry.schema.json` (URI + archive SHA256 + config/manifest/prediction-or-timing SHAs + commit + verification metadata). This PR ships an empty registry.

### Claim `measured` conditions

All of: archaeology blockers closed; real_evidence audit passed; hash-matched metrics or timing bundle; registry entry `author_verified: true`; ledger support column updated in a dedicated PR. This harness alone never promotes claims.

## Layout

- Schemas: `schemas/`
- Configs: `configs/`
- Fixtures: `fixtures/` (synthetic CSV only)
- Scripts: `scripts/iac2026/`, `scripts/benchmark_cv_core_speed.py`
- Run bundles: `results/iac2026/reproduction/<run_id>/` (gitignored)
- Policy: `ENVIRONMENT_POLICY.md`
