# Claim–evidence ledger (IAC 2026 abstract ↔ manuscript)

## Support levels

| Level | Meaning |
|-------|---------|
| `implemented` | Feature exists in code; **not** a measured scientific performance claim |
| `measured` | Ran under a documented protocol with reproducible pinned outputs |
| `accepted_abstract_reproduction_pending` | Numbers appear in the accepted IAC abstract; repo reproduction pack not yet closed |
| `protocol_defined_pending_data` | Independent evaluation contract pinned; labeled/SHA-pinned dataset not yet present |
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
| C05 | 0.894 AUROC, 0.847 AUPRC, 0.823 F1 | Detection bench | Harness (software-verification infrastructure): `scripts/iac2026/audit_reproduction_inputs.py`, `reproduce_detection_metrics.py`; contracts `reproduction/iac2026/`; archaeology `paper/iac2026/reproduction/`; run bundle `results/iac2026/reproduction/<run_id>/`; definition audit `C05_C06_DEFINITION_AUDIT.md`, `DATASET_MANIFEST_GAPS.md`, `C05_C06_DEFINITIONS.yaml`, `AUTHOR_QUESTIONNAIRE_C05_C06.md`, `C05_C06_RESPONSE_STATUS.md`, `AUTHOR_ATTESTATION_C05_C06.template.md` | T_detect | `accepted_abstract_reproduction_pending` | Harness implemented for SW verification; claim not measured; no pass/fail vs 0.894; historical defs still UNKNOWN; independent_eval_v1 is a separate current protocol and must not be reported as reproduction of these numbers |
| C06 | PaDiM/PatchCore baseline 0.856 AUROC | Same detection bench | Fail-loud stubs `scripts/iac2026/baselines/` (no anomalib; no fake average); same definition-audit artifacts as C05 | T_detect | `accepted_abstract_reproduction_pending` | Cite Defard/Roth; baseline identity still UNKNOWN; weights path unverified; see `AUTHOR_QUESTIONNAIRE_C05_C06.md` / `C05_C06_RESPONSE_STATUS.md`; PaDiM/PatchCore under independent_eval_v1 are separate runs, not a claim of 0.856 |
| C07 | 28.1 FPS @ 256×256 lightweight (no learned depth/AE) | Workstation timing | `scripts/benchmark_cv_core_speed.py` profiles: historical_exact (`historical_opencv_surrogate_8f7e3ff`) vs `current_enhancement_historical_surrogate` (supplementary surrogate profile; not the full current production pipeline; not the accepted 28.1 FPS reproduction); SW config `c07_software_verification.example.yaml` | T_hw | `accepted_abstract_reproduction_pending` | Historical harness implemented; equivalence to 28.1 pending; do not mix metric names |
| IND_EVAL_V1 | Current reproducible image-binary detection eval (protocol pin) | Independent bench | `INDEPENDENT_EVALUATION_PROTOCOL.md`, `INDEPENDENT_EVAL_V1_ANNOTATION_GUIDE.md`, `INDEPENDENT_EVAL_V1_DATASET_PLAN.md`, `INDEPENDENT_EVAL_V1_DOMAIN_SELECTION.md`, `reproduction/iac2026/INDEPENDENT_EVAL_V1.yaml` (runtime lock via `independent_eval_contract.py`), `configs/independent_evaluation.example.yaml` / `.synthetic.yaml`, SHA-pinned `manifests/independent_eval_v1.csv` (primary benchmark; freeze marker present) | T_detect_indep (planned) | `protocol_defined_pending_data` | Support value retained for compatibility. Dataset, v1_1 human review, and validation artifacts exist; headline claim closure remains pending because the frozen test split is unopened (not measured). Not a reproduction of C05/C06 accepted numbers. Supplementary human-reviewed audit is a separate row (IND_EVAL_V1_1). |
| IND_EVAL_V1_1 | Supplementary human-reviewed validation audit (frozen ARTPS scores relabeled) | Independent bench / validation only | `INDEPENDENT_EVAL_V1_1_REPORT.md`, `INDEPENDENT_EVAL_V1_1_LABEL_AUDIT.md`, `INDEPENDENT_EVAL_V1_1_EVIDENCE_MAP.md`, SHA-pinned `manifests/independent_eval_v1_1.csv` + `.meta.json`, `annotations/independent_eval_v1_1_review_provenance.csv` + `.meta.json`, frozen predictions `artps_full_frozen_mars_clf_on_v1/predictions.csv` | `tab:indep-v11` | `measured` | Validation-only supplementary relabel audit; no test result; AUROC 0.772 is the principal ranking metric; threshold-dependent F1 is not a headline metric (predeclared selection collapsed to all-positive; see audit artifacts); not a C05 reproduction; repeat-author, not independent second annotation. |
| C08 | Profile-aware onboard screening (edge-oriented) | Hardware profiles | Workstation wording; Jetson later — **not** the C07 FPS claim | T_hw / T_jetson | `planned` | Not flight HW / not flight-qualified |
| C09 | Rover-body + boundary-shadow FP suppression | Shadow / FP proxy | `scripts/run_iac_shadow_proxy_eval.py`, `false_positive_masks.py` | T_shadow_proxy | `proxy` | Mechanism also `implemented`; n=21 curated |
| C10 | Object-in-shadow protection | Shadow proxy rock-loss | same as C09 | T_shadow_proxy | `proxy` | Preliminary proxy only |
| C11 | Size–distance policy reduces field-scale FPR | Size–distance proxy | `run_iac_size_distance_proxy_eval.py`, `size_distance.py` | T_sd_proxy | `proxy` | Apparent size only; not in abstract |
| C12 | Size–distance lite bench | Unit / synthetic | `tests/test_size_distance_lite_bench.py` | none | `software_verification` | Not a performance result |
| C13 | Real-ESRGAN improves detection / quality | — | enhance profile only | — | `unsupported` | Methods one paragraph; no Results |
| C14 | Jetson / edge throughput | Jetson protocol | none yet | T_jetson | `planned` | Not flight hardware / not certification |
| C15 | Metric distance / calibrated metres / metric 3D size | — | — | — | `unsupported` | Forbidden claim |
| C16 | Depth-on-RGB visualization QC | Methods / QC | depth viz QC in `app.py` | optional Fig QC | `implemented` | Not a detection metric |

**This PR rule:** Results must not rewrite C05–C07 as final tables. Proxy work (C09–C11) stays out of the abstract; document as preliminary proxy analysis in protocol / Results stubs / Discussion / this ledger.
