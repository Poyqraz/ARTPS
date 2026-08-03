# Pinned CI deps for software-verification of the IAC reproduction harness.

## CI

```text
python -m pip install -r reproduction/iac2026/requirements-ci.txt
```

## Real evidence environment

Before any `evidence_mode: real_evidence` run that may close C05–C07:

1. Export an exact environment lock (`pip freeze` or conda `env export`) and store it next to the run bundle.
2. Record Python, OS, CPU, RAM, OpenCV, NumPy, scikit-learn, and (if used) PyTorch/CUDA versions in `environment.json`.
3. Pin dataset root via `dataset_root_env`, checkpoint SHA256, and git commit SHA with a clean tree (`allow_dirty_git: false`).
4. Register the archived run URI + SHA256 in `reproduction/iac2026/evidence_registry.json` after upload to a durable store (GitHub Release, Zenodo, or equivalent).

Software-verification outputs are never sufficient for claim closure.
