# Claim–evidence ledger (IAC 2026 abstract ↔ manuscript)

Support levels: `measured` | `proxy` | `software_verification` | `planned` | `unsupported`.

Depth language: relative depth ordering / apparent size / image-relative near–far only. Metric distance and metric 3D size are **forbidden** (`unsupported`).

Source abstract: portal / T3 ARTPS abstract (same claims as `main.tex` abstract block). Full 32p manuscript: `docs/Full_Baydemir_ARTPS.pdf` (rewrite later; do not dump).

| ID | Claim (abstract / ops) | Experiment | Script / artifact | Table / Fig | Support | Notes |
|----|------------------------|------------|-------------------|-------------|---------|-------|
| C01 | Multi-cue anomaly + monocular relative depth | Methods description | `src/models/depth_estimation.py`, fusion in `app.py` | Fig pipeline (planned) | `measured` (code) | No metric depth |
| C02 | Entropy-weighted dynamic fusion | Component / primary eval | Prior full-MS eval; reproduce runner TBD | T_detect / Fig fusion | `planned` | Need pinned script + seed |
| C03 | Soft similarity penalty + feature-space clustering | Ranking / diversity | curiosity / policy in `app.py`, `src/core/` | T_rank (planned) | `planned` | Protocol in EXPERIMENT_PROTOCOL |
| C04 | Priority Buffer second-chance re-eval | Operational FN buffer | `apply_operational_target_policy` / Priority Buffer in `app.py` | Discussion / T_ops | `planned` | Safety matrix S04 |
| C05 | 0.894 AUROC, 0.847 AUPRC, 0.823 F1 | Detection benchmark | Document exact runner + split; numbers from portal abstract | T_detect | `planned` | Do not reprint as closed until reproducible |
| C06 | PaDiM/PatchCore baseline 0.856 AUROC | Same detection bench | Same as C05 | T_detect | `planned` | Cite Defard/Roth |
| C07 | 28.1 FPS @ 256×256 lightweight (no learned depth/AE) | Workstation timing | Candidate: `scripts/benchmark_cv_core_speed.py` | T_hw | `planned` | Re-run and pin config |
| C08 | Profile-aware onboard screening (edge-oriented) | Hardware profiles | Workstation now; Jetson protocol later | T_hw / T_jetson | `planned` | Not flight HW |
| C09 | Rover-body + boundary-shadow FP suppression | Shadow / FP proxy | `scripts/run_iac_shadow_proxy_eval.py`, `src/core/false_positive_masks.py` | T_shadow_proxy | `proxy` | `results/paper_figs/iac_shadow_proxy_*` |
| C10 | Object-in-shadow protection (no shadow-rock loss) | Shadow proxy rock-loss | same as C09 | T_shadow_proxy | `proxy` | rock loss 0.0 on n=21 run |
| C11 | Size–distance policy reduces field-scale FPR | Size–distance proxy | `scripts/run_iac_size_distance_proxy_eval.py`, `src/core/size_distance.py` | T_sd_proxy | `proxy` | apparent size only |
| C12 | Size–distance lite bench | Unit / synthetic | `tests/test_size_distance_lite_bench.py` | none | `software_verification` | Not a performance result |
| C13 | Real-ESRGAN improves detection / quality | — | enhance profile only | — | `unsupported` | Methods one paragraph; no Results |
| C14 | Jetson / flight-like edge throughput | Jetson protocol | none yet | T_jetson | `planned` | See EXPERIMENT_PROTOCOL |
| C15 | Metric distance / calibrated metres / metric 3D size | — | — | — | `unsupported` | Forbidden claim |
| C16 | Depth-on-RGB visualization QC | Methods / QC | depth viz QC in `app.py` | optional Fig QC | `measured` (code) | Not a detection metric |

**This PR rule:** Results section must not rewrite C05–C07 as final tables; keep stubs + this ledger.
