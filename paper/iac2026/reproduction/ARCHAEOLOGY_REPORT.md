# IAC archaeology report — accepted abstract C05–C07

Salt-okunur git/PDF taraması. Tahmin yok. Bilinmeyenler: `UNKNOWN — evidence not located`.

Ledger claim IDs (locked): **C05** metrics · **C06** baseline 0.856 · **C07** 28.1 FPS · **C08** profile-aware screening (planned, not FPS).

## Claim evidence table

| Claim | Located source | Commit/path | Dataset | Split | Task level | Score definition | Threshold policy | Model/config | Reproducible now? | Missing information |
|------|----------------|-------------|---------|-------|------------|------------------|------------------|--------------|-------------------|--------------------|
| C05 0.894 AUROC / 0.847 AUPRC / 0.823 F1 | Accepted abstract PDFs; `paper/iac2026/main.tex`; Full MS Table 2 narrative | Abstract PDFs under `docs/` (often untracked); ledger since `457c91f`+; Full: `docs/Full_Baydemir_ARTPS.pdf`; historical `docs/paper.tex` @ `8f7e3ff`; charts hardcoded in `scripts/generate_paper_charts.py` @ `8f7e3ff` | Full MS: 2847 images (Curiosity Mastcam 1247 Sol 100–1700; Perseverance Mastcam-Z 892 Sol 1–400); test/val pool mentioned as 708 | File list / seed **UNKNOWN** | **UNKNOWN** (image/pixel/region not stated) | “anomaly discrimination” scores; orientation assumed higher=more anomalous but not pinned | “Decision thresholds are selected on validation data; results on the test set” — **numeric threshold UNKNOWN** | ARTPS hybrid stack (narrative) | **No** — no pinned predictions or eval runner on HEAD | task_level; positive_label; score formula; split CSV; seed; F1 threshold value; raw predictions |
| C06 0.856 AUROC PaDiM/PatchCore | Same Table 2 / abstract baseline | Full PDF; `8f7e3ff` paper.tex; weights `results/padim_stats.pth`, `results/patchcore_bank.pth` (local, gitignored) | Same as C05 (assumed) | **UNKNOWN** | **UNKNOWN** | Mahalanobis / PatchCore distance maps described in Methods; aggregation to image score **UNKNOWN** | **UNKNOWN** | Labeled “PaDiM/PatchCore (WRN-50-2)” as **one** column — PaDiM alone vs PatchCore alone vs combo **UNKNOWN** | **No** — weights loadable in app; no metric runner producing 0.856 on HEAD | baseline identity; extractor version; train bank recipe; eval script |
| C07 28.1 FPS @ 256×256 core-only | Abstract + Full MS hardware profile | Abstract text; historical `scripts/benchmark_cv_core_speed.py` @ `8f7e3ff` / stash (OpenCV-only, no Torch) | Synthetic/random or local images in historical script | N/A | N/A (timing) | FPS = throughput of lightweight OpenCV enhance+fusion+localize path | N/A | Claims: exclude learned depth & AE; workstation | **Partial harness only** — script not on HEAD; **no pinned 28.1 raw timing JSON** | host CPU/GPU; commit SHA of 28.1 run; warm-up/timed counts; whether decode/resize included in headline |
| C08 profile-aware onboard screening | Ledger / protocol only | `CLAIM_EVIDENCE_LEDGER.md` | N/A | N/A | N/A | N/A | N/A | Jetson planned separately | **No** (planned) | Jetson device + run |

## Answers to required questions

1. **AUROC image / pixel / region?** `UNKNOWN — evidence not located` (paper says “anomaly discrimination”; no explicit aggregation level).
2. **AUPRC labels and scores?** `UNKNOWN — evidence not located` beyond “anomaly” positive class narrative.
3. **F1 threshold selection?** Validation-set selection stated in Full MS; **numeric policy and threshold value UNKNOWN**.
4. **Positive class?** Narratively anomalous / scientifically interesting targets; **operational label definition UNKNOWN**.
5. **0.856 single baseline, average, or best?** Reported as one “PaDiM/PatchCore (WRN-50-2)” column → **UNKNOWN** whether single, max, or mean of two.
6. **Dataset split images?** Counts known (2847 / 1247 / 892 / 708); **per-file split membership UNKNOWN**.
7. **Same Sol/scene across splits?** `UNKNOWN — evidence not located` (no `scene_group_id` manifest found).
8. **28.1 FPS includes decode/resize/preprocess/postprocess?** Historical script times a combined OpenCV pipeline; **headline stage breakdown for the published 28.1 UNKNOWN**.
9. **CPU/GPU / software env for 28.1?** “development workstation” only → **UNKNOWN**.
10. **Raw prediction or timing records?** `UNKNOWN — evidence not located` (no pinned `predictions.csv` / `timing_raw.csv` for accepted numbers).

## HEAD blockers (must close before real C05/C06 run)

- task_level  
- positive_label / label_semantics  
- score definition + orientation  
- dataset split + seed + SHA256-pinned manifest  
- threshold_policy (fixed historical **or** validation-only selection)  
- baseline identity for C06  

**Stop rule:** harness may compute metrics from synthetic or future pinned prediction tables; do **not** start real NASA inference claiming C05/C06 closure until blockers above are evidence-backed.

## Related HEAD assets (not sufficient for closure)

- Proxy evals: `scripts/run_iac_shadow_proxy_eval.py`, `run_iac_size_distance_proxy_eval.py` — **not** C05/C06 evidence.
- App fusion loads PaDiM/PatchCore for visualization — not an AUROC protocol.
- `src/eval/metrics.py` / prep tools / speed bench exist historically (`8f7e3ff`) but are **absent from current HEAD** (restore via new paths only; no legacy branch merge).
