# IAC archaeology report — accepted abstract C05–C07

Salt-okunur git/PDF taraması. Tahmin yok. Bilinmeyenler: `UNKNOWN — evidence not located`.

Ledger claim IDs (locked): **C05** metrics · **C06** baseline 0.856 · **C07** 28.1 FPS · **C08** profile-aware screening (planned, not FPS).

## Source types

| Tag | Meaning |
|-----|---------|
| accepted abstract claim | Numbers in accepted IAC abstract / portal PDFs |
| manuscript narrative | Full MS prose / tables without raw artifacts |
| historical source code | Recoverable from a git commit path |
| hard-coded visualization value | Chart/script constant (not a raw experiment output) |
| raw experiment artifact | Pinned predictions / timing CSV with provenance |
| unverified local path | Path referenced in code; presence/checksum not verified here |

Hard-coded values in historical `scripts/generate_paper_charts.py` are **hard-coded visualization values**, not raw experiment artifacts.

## Claim evidence table

| Claim | Located source | Source type | Commit/path | Dataset | Split | Task level | Score definition | Threshold policy | Model/config | Reproducible now? | Missing information |
|------|----------------|-------------|-------------|---------|-------|------------|------------------|------------------|--------------|-------------------|--------------------|
| C05 0.894 AUROC / 0.847 AUPRC / 0.823 F1 | Accepted abstract PDFs; `paper/iac2026/main.tex`; Full MS Table 2 | accepted abstract claim; manuscript narrative; hard-coded visualization value | Abstract PDFs under `docs/` (often untracked); Full: `docs/Full_Baydemir_ARTPS.pdf`; historical `docs/paper.tex` @ `8f7e3ff`; charts @ `8f7e3ff` | Full MS counts: 2847 images (Curiosity Mastcam 1247 Sol 100–1700; Perseverance Mastcam-Z 892 Sol 1–400); 708 mentioned as test/val pool | File list / seed **UNKNOWN** | **UNKNOWN** | “anomaly discrimination”; orientation not pinned | Validation-selected narrative; numeric threshold **UNKNOWN** | ARTPS hybrid stack (narrative) | **No** | task_level; labels; score formula; split CSV; F1 threshold; raw predictions |
| C06 0.856 AUROC PaDiM/PatchCore | Same Table 2 / abstract baseline | accepted abstract claim; manuscript narrative; unverified local path | Full PDF; `8f7e3ff` paper.tex; code references `results/padim_stats.pth`, `results/patchcore_bank.pth` | Dataset linkage not independently recoverable; values appear in the same manuscript table as C05 | **UNKNOWN** | **UNKNOWN** | Aggregation to image score **UNKNOWN** | **UNKNOWN** | “PaDiM/PatchCore (WRN-50-2)” single column — identity **UNKNOWN** | **No** — code references expected local weight paths; presence/checksum/provenance unverified; no metric runner producing 0.856 on HEAD | baseline identity; bank recipe; eval script |
| C07 28.1 FPS @ 256×256 core-only | Abstract + Full MS hardware profile | accepted abstract claim; historical source code | historical `scripts/benchmark_cv_core_speed.py` @ `8f7e3ff` (OpenCV-only). Stash-only copies without exported hash are **not** treated as reproducible sources | N/A | N/A | N/A | Headline stage for published 28.1 **UNKNOWN** | N/A | Exclude learned depth & AE; workstation | Partial harness on HEAD (`historical_opencv_surrogate_8f7e3ff`); **no pinned 28.1 timing JSON** | host; commit of 28.1 run; warm-up/timed counts |
| C08 profile-aware onboard screening | Ledger / protocol | manuscript narrative | `CLAIM_EVIDENCE_LEDGER.md` | N/A | N/A | N/A | N/A | N/A | Jetson planned | **No** (planned) | device + run |

## Dataset count ambiguity

Manuscript numbers 1247 (Curiosity), 892 (Perseverance), and 708 (test/validation pool) do **not** fully specify source-vs-split membership. Without a per-file manifest, the 2847/1247/892/708 structure cannot be reconstructed.

## Answers to required questions

1. **AUROC image / pixel / region?** `UNKNOWN — evidence not located`
2. **AUPRC labels and scores?** `UNKNOWN — evidence not located` (method also UNKNOWN: average_precision vs trapezoidal_pr_auc)
3. **F1 threshold selection?** Validation-set selection stated; **numeric policy/value UNKNOWN**
4. **Positive class?** Operational label definition **UNKNOWN**
5. **0.856 single / average / best?** **UNKNOWN**
6. **Dataset split images?** Counts known; **per-file membership UNKNOWN**
7. **Same Sol/scene across splits?** `UNKNOWN — evidence not located`
8. **28.1 FPS includes decode/resize/...?** **UNKNOWN**
9. **CPU/GPU / software env for 28.1?** **UNKNOWN**
10. **Raw prediction or timing records?** `UNKNOWN — evidence not located`

## HEAD blockers (must close before real C05/C06 run)

- task_level
- positive_label / label_semantics
- score definition + orientation
- dataset split + seed + SHA256-pinned manifest
- threshold_policy (fixed historical **or** validation-only selection)
- baseline identity for C06
- pr_metric_method for accepted 0.847

**Stop rule:** harness may compute metrics from synthetic or future pinned prediction tables; do **not** start real NASA inference claiming C05/C06 closure until blockers above are evidence-backed.

## Follow-up audit (definitions branch)

Deeper salt-okunur scan + author questionnaire (no invented splits/labels):

- [C05_C06_DEFINITION_AUDIT.md](C05_C06_DEFINITION_AUDIT.md)
- [DATASET_MANIFEST_GAPS.md](DATASET_MANIFEST_GAPS.md)
- [AUTHOR_QUESTIONNAIRE_C05_C06.md](AUTHOR_QUESTIONNAIRE_C05_C06.md)
- [`reproduction/iac2026/C05_C06_DEFINITIONS.yaml`](../../../reproduction/iac2026/C05_C06_DEFINITIONS.yaml)
- Readiness: `python scripts/iac2026/check_c05_c06_definition_readiness.py` → expect `real_run_allowed=false` until P0 closed.

## Related HEAD assets (not sufficient for closure)

- Proxy evals: `scripts/run_iac_shadow_proxy_eval.py`, `run_iac_size_distance_proxy_eval.py` — **not** C05/C06 evidence.
- App fusion loads PaDiM/PatchCore for visualization — not an AUROC protocol.
- Inference — app may load weights if present locally; that is not a closed detection protocol.
- Historical metrics/speed code recovered into `scripts/iac2026/` and `scripts/benchmark_cv_core_speed.py` (new paths only).
