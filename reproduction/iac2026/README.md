# IAC 2026 reproduction harness

Evidence software for claims **C05–C07**. Manuscript stays under `paper/iac2026/`. This tree does **not** promote accepted-abstract numbers to measured results.

## Workflow

1. **Archaeology** — read `paper/iac2026/reproduction/ARCHAEOLOGY_REPORT.md` and `REPRODUCTION_STATUS.md`.
2. **Manifest** — pin a dataset CSV matching `schemas/dataset_manifest.schema.json`.
3. **Audit** — `python scripts/iac2026/audit_reproduction_inputs.py --config ...` (fail closed).
4. **Metrics** — `python scripts/iac2026/reproduce_detection_metrics.py --config ...` from a **prediction table** (no test-set threshold search).
5. **Baselines** — adapters under `scripts/iac2026/baselines/` (fail-loud until contracts known).
6. **Speed (C07)** — `python scripts/benchmark_cv_core_speed.py --config ...` (core-only; report actual FPS).

## When `measured` is allowed

Only after archaeology closes: `task_level`, positive label, score definition, train/val/test file list (or seed), threshold policy, and (for C06) baseline identity. Until then ledger stays `accepted_abstract_reproduction_pending`. Synthetic fixtures = **software verification only**.

## Example configs

- `configs/detection_reproduction.example.yaml` — ships with `task_level: TASK_LEVEL_TBD` and `threshold_policy: unknown` so real C05/C06 inference stays blocked.
- `configs/core_speed_256.example.yaml` — C07 workstation core-only timing.

## Fixtures

Synthetic CSV only. No NASA images or model weights in this tree.
