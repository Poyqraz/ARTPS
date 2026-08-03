# Claim–evidence ledger (IAC 2026 abstract ↔ manuscript)

## Support levels

| Level | Meaning |
|-------|---------|
| `implemented` | Feature exists in code; **not** a measured scientific performance claim |
| `measured` | Ran under a documented protocol with reproducible pinned outputs |
| `accepted_abstract_reproduction_pending` | Numbers appear in the accepted IAC abstract; repo reproduction pack not yet closed |
| `proxy` | Evaluation uses class/mask/OFF-run pseudo-GT — **not** human bbox GT |
| `software_verification` | Unit / synthetic / regression check; not a performance result |
| `planned` | Experiment defined but not executed |
| `unsupported` | Must not be claimed with current evidence |

Depth language: relative depth ordering / apparent size / image-relative near–far only. Metric distance and metric 3D size are **forbidden** (`unsupported`).

Source: accepted IAC abstract (same numerical claims as `main.tex`). Full 32p MS: `docs/Full_Baydemir_ARTPS.pdf` (rewrite later; do not dump).

| ID | Claim (abstract / ops) | Experiment | Script / artifact | Table / Fig | Support | Notes |
|----|------------------------|------------|-------------------|-------------|---------|-------|
| C01 | Multi-cue anomaly + monocular relative depth | Methods | `src/models/depth_estimation.py`, fusion in `app.py` | Fig pipeline (planned) | `implemented` | No metric depth |
| C02 | Entropy-weighted dynamic fusion | Methods / future ablation | fusion code in `app.py`; pinned eval TBD | T_detect / Fig fusion | `implemented` | Performance delta still `planned` |
| C03 | Soft similarity + feature-space clustering | Ranking | curiosity / policy in `app.py`, `src/core/` | T_rank (planned) | `implemented` | Quantitative ranking study `planned` |
| C04 | Priority Buffer second-chance re-eval | Operational FN buffer | Priority Buffer / operational policy in `app.py` | Discussion / T_ops | `implemented` | Quantitative buffer study `planned` (S04) |
| C05 | 0.894 AUROC, 0.847 AUPRC, 0.823 F1 | Detection bench | Pin runner + split; accepted abstract | T_detect | `accepted_abstract_reproduction_pending` | Do not ship as final Results table yet |
| C06 | PaDiM/PatchCore baseline 0.856 AUROC | Same detection bench | Same as C05 | T_detect | `accepted_abstract_reproduction_pending` | Cite Defard/Roth |
| C07 | 28.1 FPS @ 256×256 lightweight (no learned depth/AE) | Workstation timing | Candidate: `scripts/benchmark_cv_core_speed.py` | T_hw | `accepted_abstract_reproduction_pending` | Re-run and pin config |
| C08 | Profile-aware onboard screening (edge-oriented) | Hardware profiles | Workstation wording; Jetson later | T_hw / T_jetson | `planned` | Not flight HW / not flight-qualified |
| C09 | Rover-body + boundary-shadow FP suppression | Shadow / FP proxy | `scripts/run_iac_shadow_proxy_eval.py`, `false_positive_masks.py` | T_shadow_proxy | `proxy` | Mechanism also `implemented`; n=21 curated |
| C10 | Object-in-shadow protection | Shadow proxy rock-loss | same as C09 | T_shadow_proxy | `proxy` | Preliminary proxy only |
| C11 | Size–distance policy reduces field-scale FPR | Size–distance proxy | `run_iac_size_distance_proxy_eval.py`, `size_distance.py` | T_sd_proxy | `proxy` | Apparent size only; not in abstract |
| C12 | Size–distance lite bench | Unit / synthetic | `tests/test_size_distance_lite_bench.py` | none | `software_verification` | Not a performance result |
| C13 | Real-ESRGAN improves detection / quality | — | enhance profile only | — | `unsupported` | Methods one paragraph; no Results |
| C14 | Jetson / edge throughput | Jetson protocol | none yet | T_jetson | `planned` | Not flight hardware / not certification |
| C15 | Metric distance / calibrated metres / metric 3D size | — | — | — | `unsupported` | Forbidden claim |
| C16 | Depth-on-RGB visualization QC | Methods / QC | depth viz QC in `app.py` | optional Fig QC | `implemented` | Not a detection metric |

**This PR rule:** Results must not rewrite C05–C07 as final tables. Proxy work (C09–C11) stays out of the abstract; document as preliminary proxy analysis in protocol / Results stubs / Discussion / this ledger.
