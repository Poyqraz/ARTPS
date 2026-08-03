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

Document the exact image list and seed when closing C05–C07. Current proxy runs used curated benchmark subsets (e.g. round3 + rover samples, `n=21` in shadow/size-distance proxy summaries). Pin paths in the ledger when re-run.

## Baselines

- PaDiM / PatchCore-style anomaly maps (cite Defard, Roth).
- Heuristic contour + fusion baseline for proposal metrics (`results/paper_figs/paper_report.md` axes).

## Metrics (names only in manuscript)

AUROC, AUPRC, F1, FPS, latency, peak memory; proxy FPR / recall / avg detections; self-IoU for merge stability. **No textbook formula expansions.**

## Ablation protocol

- Toggle gates: FP suppression ON/OFF; size–distance policy ON/OFF (`--fp_mode`, `--size_distance_policy` on export / proxy runners).
- Record config JSON next to each `results/paper_figs/iac_*_summary.json`.

## Proxy eval (not human bbox GT)

| Study | Script | Artifacts |
|-------|--------|-----------|
| Shadow / FP | `scripts/run_iac_shadow_proxy_eval.py` | `results/paper_figs/iac_shadow_proxy_*` |
| Size–distance | `scripts/run_iac_size_distance_proxy_eval.py` | `results/paper_figs/iac_size_distance_proxy_*` |
| Lite size–distance | `tests/test_size_distance_lite_bench.py` | software verification only |

## Workstation timing (C07)

Candidate: `scripts/benchmark_cv_core_speed.py` (lightweight OpenCV + fusion + localization; exclude learned depth/AE). Pin resolution (256 / 384 / 768), device, commit SHA.

## Jetson benchmark protocol (C14) — planned, not executed

**Goal:** profile-aware FPS/latency/memory on NVIDIA Jetson (Orin Nano / Orin NX or equivalent). Not flight certification.

1. **Images:** fixed list (≥20) from the detection bench; same preprocess as workstation.
2. **Profiles:** (a) lightweight core only; (b) + relative depth; (c) + AE/learned anomaly if memory allows.
3. **Resolutions:** 256, 384, 768 (square letterbox as in app).
4. **Metrics:** mean/p95 latency, FPS, peak RSS / GPU mem, thermal throttle flags.
5. **Runs:** 30 warm + 100 timed; report mean ± std; record JetPack / CUDA / Power mode.
6. **Out of scope:** claiming flight readiness; metric depth.

Deliverable when run: `results/paper_figs/iac_jetson_summary.json` + table T_jetson.
