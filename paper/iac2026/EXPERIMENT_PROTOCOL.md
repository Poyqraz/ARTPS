# Experiment protocol (dataset, ablation, proxy, Jetson)

Authoritative for `sections/experiments.tex`. Bind claims via [`CLAIM_EVIDENCE_LEDGER.md`](CLAIM_EVIDENCE_LEDGER.md).

## Dataset

- **Source:** NASA Mars rover imagery used by ARTPS (Curiosity / Perseverance field images in local `mars_images/` and benchmark sets under `results/benchmark_*` where present).
- **Task labels:** terrain / object class labels used by the app and proxy evals (e.g. rocky, rover, dusty). Not a public COCO-style bbox pack unless separately published.
- **Depth:** monocular relative depth maps from DPT_Large or fallback CNN — within-image ordering only ([`DEPTH_PERCEPTION_TODO.md`](../../DEPTH_PERCEPTION_TODO.md), [`results/paper_figs/depth_semantics.md`](../../results/paper_figs/depth_semantics.md)).

## Annotation / labeling

| Layer | What | Used for |
|-------|------|----------|
| Class / terrain labels | Existing category tags | Curiosity / known-value terms; proxy class filters |
| Human bbox GT | Not required for proxy milestone | Future localization paper tables |
| OFF-run pseudo-GT | Detections from policy/gates OFF | Proxy recall / FPR deltas |
| Masks | Rover-body, shadow, depth edges | FP suppression proxies |

## Splits

Document the exact image list and seed when closing C05–C07 (`accepted_abstract_reproduction_pending`). Current proxy runs used curated benchmark subsets (e.g. round3 + rover samples, `n=21` in shadow/size-distance proxy summaries). Pin paths in the ledger when re-run.

Manifest / prediction contracts: `reproduction/iac2026/schemas/`. Archaeology blockers: [`reproduction/ARCHAEOLOGY_REPORT.md`](reproduction/ARCHAEOLOGY_REPORT.md). Status table: [`reproduction/REPRODUCTION_STATUS.md`](reproduction/REPRODUCTION_STATUS.md).

## Baselines

- PaDiM / PatchCore-style anomaly maps (cite Defard, Roth).
- Harness adapters (fail-loud until contract known): `scripts/iac2026/baselines/padim_adapter.py`, `patchcore_adapter.py`. Do **not** average into a fake 0.856; do **not** use anomalib for the C06 column.
- Heuristic contour + fusion axes listed in `results/paper_figs/paper_report.md` are **exploratory only** (sample count = 5; not a quantitative manuscript result).

## Metrics (names only in manuscript)

AUROC, AUPRC, F1, FPS, latency, peak memory; proxy FPR / recall / avg detections; self-IoU for merge stability. **No textbook formula expansions.**

Prediction-table runner (software verification / future pinned preds): `scripts/iac2026/reproduce_detection_metrics.py`. Input audit (fail closed): `scripts/iac2026/audit_reproduction_inputs.py`. Artifacts under `results/iac2026/reproduction/<run_id>/`. Never pass/fail against accepted-abstract targets.

## Ablation protocol

- Toggle gates: FP suppression ON/OFF; size–distance policy ON/OFF (`--fp_mode`, `--size_distance_policy` on export / proxy runners).
- Record config JSON next to each `results/paper_figs/iac_*_summary.json`.

## Preliminary proxy analysis (not human bbox GT; not in abstract)

Proxy studies are **preliminary** on a curated `n=21` set. They must not be presented as accepted-abstract headline results.

| Study | Script | Artifacts | Support |
|-------|--------|-----------|---------|
| Shadow / FP | `scripts/run_iac_shadow_proxy_eval.py` | `results/paper_figs/iac_shadow_proxy_*` | `proxy` |
| Size–distance | `scripts/run_iac_size_distance_proxy_eval.py` | `results/paper_figs/iac_size_distance_proxy_*` | `proxy` |
| Lite size–distance | `tests/test_size_distance_lite_bench.py` | — | `software_verification` |

## Workstation timing (C07)

Accepted abstract: 28.1 FPS at 256×256 in a lightweight configuration **excluding learned depth and AE inference** (`accepted_abstract_reproduction_pending`).

Harness: `scripts/benchmark_cv_core_speed.py` (OpenCV core via `scripts/iac2026/cv_core_pipeline.py`). Example config: `reproduction/iac2026/configs/core_speed_256.example.yaml`. Requires depth off, AE off, 256×256, batch 1, warm-up ≥30, timed ≥300; headline = `core_processing` FPS. Pin whatever the machine produces — do **not** claim match to 28.1 until a closed measured run is ledgered.

## Jetson benchmark protocol (C14) — planned, not executed

**Status:** `planned`. **Not** flight hardware. **Not** flight certification.

**Exact target device:** to be fixed before execution (do not report Orin Nano / Orin NX as a scientific result until author confirms hardware).

### Required log fields (before/during run)

| Field | Notes |
|-------|-------|
| Exact target device | TBD until author confirms |
| JetPack version | record string |
| CUDA / cuDNN / TensorRT versions | record strings |
| `nvpmodel` power mode | record mode id/name |
| `jetson_clocks` | enabled / disabled |
| Batch size | typically 1 for onboard screening |
| Precision | FP32 / FP16 / INT8 |
| Backend | PyTorch / ONNX Runtime / TensorRT |
| Mean latency | ms |
| p95 latency | ms |
| p99 or worst-case latency | ms |
| FPS | derived or measured |
| Peak RAM / GPU memory | MB |
| Average power | W |
| Peak power | W |
| Energy per frame | J/frame |
| CPU / GPU temperature | °C |
| Thermal throttling | yes/no + events |
| Warm-up count | e.g. 30 |
| Timed-run count | e.g. 100 |
| Commit SHA | git revision |
| Config file | path to pinned JSON/YAML |

### Procedure sketch

1. **Images:** fixed list (≥20) from the detection bench; same preprocess as workstation.
2. **Profiles:** (a) lightweight core only; (b) + relative depth; (c) + AE/learned anomaly if memory allows.
3. **Resolutions:** 256, 384, 768 (square letterbox as in app).
4. **Out of scope:** flight certification claims; metric distance.

Deliverable when run: `results/paper_figs/iac_jetson_summary.json` + table T_jetson.
