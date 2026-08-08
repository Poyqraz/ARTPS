# Legacy extended-manuscript asset audit (IAC 2026)

Status: `LEGACY_ASSET_AUDIT_COMPLETE` (research + classification only).
Branch: `paper/iac2026-legacy-asset-audit` from `main` @ `5f4ca2b` (PR #33).
Mode: **no TeX / no figure generation / no inference / no SUBMISSION_FREEZE.md**.
Decision gate: evidence decides A / B / C (not richness). **Author review required before any manuscript figure change.**

Ledger statuses confirmed unchanged:

- C05 / C06 / C07: `accepted_abstract_reproduction_pending`
- `IND_EVAL_V1`: `protocol_defined_pending_data` (`test_opened=false`, `final_test_authorized=false`)
- `IND_EVAL_V1_1`: `measured` validation-only supplementary (AUROC 0.772 / AP 0.956)

Reuse, do not re-litigate: [`reproduction/ARCHAEOLOGY_REPORT.md`](reproduction/ARCHAEOLOGY_REPORT.md), [`reproduction/C05_C06_DEFINITION_AUDIT.md`](reproduction/C05_C06_DEFINITION_AUDIT.md), [`CLAIM_EVIDENCE_LEDGER.md`](CLAIM_EVIDENCE_LEDGER.md), [`FIGURE_TABLE_PLAN.md`](FIGURE_TABLE_PLAN.md).

[`results/paper_figs/paper_report.md`](../../results/paper_figs/paper_report.md) is exploratory / top-k curiosity — **not** an IAC Results source and **not** a cherry-pick seed.

---

## 0. Manuscript lineage (internal only — not for the scientific paper)

| State | Date / commit | Source | Role |
|-------|---------------|--------|------|
| **A** | arXiv 2509.00042 (2025-08-23) | Older preprint title *Depth-Enhanced Hybrid Anomaly Detection and Learnable Curiosity Score…*; abstract ~AUROC 0.94 / AUPRC 0.89 / F1 0.87 | **Do not import** into IAC |
| **B** | Extended ARTPS MS (Jan 2026 lineage) | Local [`docs/Full_Baydemir_ARTPS.pdf`](../../docs/Full_Baydemir_ARTPS.pdf) (32 pp, untracked); TeX `8f7e3ff:docs/paper.tex`; qualitative blobs `e4282a3:ARTPS_repo/*` | Accepted-IAC numerical lineage **plus** extra figs/tables |
| **C** | IAC 2026 after PR #33 | [`paper/iac2026/`](.) 7 pp; TikZ Fig.~1; Table 1 historical; Table 2s supplementary | **Canonical submission manuscript** |

Chronology: arXiv 2025 → extended ARTPS MS Jan 2026 → accepted IAC abstract lineage → evidence-gated IAC manuscript (PR #30–#33).

January-2026 PDF output must **not** be described as current-code output: current frozen eval uses fixed-weight fusion + `max_valid_candidate_after_masks`; curiosity / diversity / Priority Buffer are UI ranking, not v1_1 image score; depth is relative only.

---

## 1. Provenance search summary

| Location | Found? | Note |
|----------|--------|------|
| `8f7e3ff:docs/paper.tex` | yes | Full figure `\includegraphics` list + Tables 2–4 numbers |
| `e4282a3:ARTPS_repo/*.jpg,*.png` | yes (git blob only) | Qualitative Figs 1–6 source files; **not** present on HEAD working tree (`ARTPS_repo/` absent) |
| `8f7e3ff:scripts/generate_paper_charts.py` | yes | Fig 7/8 **hard-coded visualization constants**, not raw run artifacts |
| `HEAD:scripts/generate_paper_charts.py` | **no** | Removed from current tree |
| `8f7e3ff:results/paper_figs/summary.csv` | yes | 5 `results/paper_images/…` rows — **different filenames** from Figs 1–6 |
| `HEAD:results/paper_images/` | **no** | Not in tree |
| `docs/docs_second/figures/fig_benchmark_grouped_bar.pdf` + `fig_ablation_drop.pdf` | local untracked | Matches hard-coded chart script |
| PDS product ID / SHA manifest for Figs 1–6 | **no** | Informal names (`curiosity_boulders.jpg`, `sand_rock.jpg`, …) |
| Checkpoint / config / prediction CSV linking those images to C05 or v1_1 | **no** | |
| Independent-eval test split used for any listed qualitative file | not established | Do **not** run ARTPS on test regardless |

SHA256 (first 16 hex) of recoverable `e4282a3:ARTPS_repo/` blobs (computed this audit; **not** a historical run record):

| File | SHA256[:16] | Bytes |
|------|-------------|-------|
| `non_enhanced.jpg` | `4577b1d1fe192adf` | 335402 |
| `enhanced.jpg` | `43a763f0cb5d2bad` | 545933 |
| `curiosity_hills_small_objects.jpg` | `b2e44d75331cb74d` | 44337 |
| `ae_diff_norm_hills_combined.png` | `f21118b069dad596` | 2256983 |
| `Mars_Perseverance_Rover_Sands.png` | `de59f24518cb7b76` | 2142432 |
| `Mars_Perseverance_Rover_Sands_depth_analsys2.png` | `13ccecefcd7a9765` | 162717 |
| `curiosity_boulders.jpg` | `96f7c830e4f8341b` | 68059 |
| `curiosity_rocks.jpg` | `1ef978081207d0ab` | 78200 |
| `curiosity_ripples_and_outcrops_hard.jpg` | `90b9f7a016ade020` | 82242 |
| `curiosity_ripples_and_outcrops_depth_anomaly.png` | `6c2c38ad56d9717c` | 1128277 |
| `sand_rock.jpg` | `df3c82719f99ee54` | 206747 |
| `combined_anomaly_detection.jpg` | `70d6346a24eb97ef` | 175326 |

TeX name `Mars_Perseverance_Rover_Sands_depth_analsys.png` vs git blob `…_analsys2.png`: **filename mismatch**.

---

## 2. Asset inventory

Classification vocabulary: `CURRENT_CODE_FAITHFUL` | `HISTORICAL_QUALITATIVE_ARTIFACT` | `HISTORICAL_REPORTED_ONLY` | `REGENERATABLE_FROM_CURRENT_ARTIFACTS` | `PROVENANCE_INCOMPLETE` | `UNSUPPORTED_FOR_CURRENT_IAC`.

### 2.1 Qualitative figures (State B Figs 1–6)

#### FIG-B1 — Input enhancement before / after

| Field | Value |
|-------|--------|
| asset_id | FIG-B1 |
| PDF page | 6 |
| historical caption | Input enhancement: dehazing and photometric adjustment clarify rover/surface boundaries. (a) Hazy/raw (b) After dehazing + enhancement |
| scientific purpose | Show DHE / photometric preprocess |
| historical values | none |
| historical source file located? | yes — `e4282a3:ARTPS_repo/non_enhanced.jpg`, `enhanced.jpg` |
| raw input located? | informal jpg only; no PDS ID |
| generation script located? | narrative CLAHE/gamma/DHE in TeX; **not** pinned to `enhance_image_auto` / Real-ESRGAN-off default |
| input SHA256 located? | computed now from git blob; **not** in a historical manifest |
| model/checkpoint linkage located? | no |
| config linkage located? | no |
| prediction/run artifact located? | no |
| current-code equivalent exists? | optional enhancement path exists (C13 `unsupported` for detection gain; Real-ESRGAN default off) |
| claim layer | State B methods illustration |
| evidence status | `PROVENANCE_INCOMPLETE` (blob exists; process ≠ proven current path) |
| recommended IAC action | **DEFER** — do not present as current ARTPS |

#### FIG-B2 — Raw + reconstruction / hybrid anomaly

| Field | Value |
|-------|--------|
| asset_id | FIG-B2 |
| PDF page | 11 |
| historical caption | (a) raw scene with far-field detail and small near-field objects. (b) hybrid anomaly fusion visualization combining normalized AE difference, image cues, and depth discontinuity cues |
| scientific purpose | Qualitative fusion output |
| historical values | none |
| historical source file located? | `curiosity_hills_small_objects.jpg`, `ae_diff_norm_hills_combined.png` @ `e4282a3` |
| raw input / SHA / script / checkpoint / config / run | informal name; SHA now-computed only; no run CSV |
| current-code equivalent exists? | AE recon + fixed-weight fusion exist; **not** proven identical to this PNG |
| claim layer | State B methods |
| evidence status | `HISTORICAL_QUALITATIVE_ARTIFACT` + `PROVENANCE_INCOMPLETE` |
| recommended IAC action | **DEFER** unless author later approves explicitly labeled historical reuse (decision C) |

#### FIG-B3 — Raw + relative-depth / discontinuity

| Field | Value |
|-------|--------|
| asset_id | FIG-B3 |
| PDF page | 12 |
| historical caption | (a) raw scene for depth analysis. (b) depth analysis and discontinuity detection (rover body / near-field geometry) |
| scientific purpose | Depth-as-geometry vs photometry; rover self-mask intuition |
| historical values | none (must not be read as metric depth accuracy) |
| historical source file located? | `Mars_Perseverance_Rover_Sands.png`; depth panel git name `…_analsys2.png` vs TeX `…_analsys.png` |
| current-code equivalent exists? | `DPT_Large` relative depth + C16 viz QC `implemented`; metric depth `unsupported` (C15) |
| claim layer | State B methods |
| evidence status | `PROVENANCE_INCOMPLETE` |
| recommended IAC action | **DEFER** — do not weaken relative-depth claim |

#### FIG-B4 — Different anomaly-component examples

| Field | Value |
|-------|--------|
| asset_id | FIG-B4 |
| PDF page | 14 |
| historical caption | Different anomaly components emphasize different target types: two example scenes (boulders / rock pile) |
| historical source file located? | `curiosity_boulders.jpg`, `curiosity_rocks.jpg` @ `e4282a3` |
| evidence status | `PROVENANCE_INCOMPLETE` |
| recommended IAC action | **DEFER** |

#### FIG-B5 — Depth–anomaly overlay

| Field | Value |
|-------|--------|
| asset_id | FIG-B5 |
| PDF page | 15 |
| historical caption | Challenging ripples/outcrops; depth–anomaly overlay after NMS/IoU merge |
| historical source file located? | `curiosity_ripples_and_outcrops_hard.jpg`, `…_depth_anomaly.png` @ `e4282a3` |
| evidence status | `PROVENANCE_INCOMPLETE` |
| recommended IAC action | **DEFER** |

#### FIG-B6 — Localization / box merging

| Field | Value |
|-------|--------|
| asset_id | FIG-B6 |
| PDF page | 16 |
| historical caption | (a) raw input for localization. (b) detected anomaly regions and box hypotheses after hybrid fusion |
| historical source file located? | `sand_rock.jpg`, `combined_anomaly_detection.jpg` @ `e4282a3` |
| current-code equivalent exists? | localization + operational masks exist; overlay ≠ frozen `max_valid_candidate_after_masks` contract |
| evidence status | `HISTORICAL_QUALITATIVE_ARTIFACT` + `PROVENANCE_INCOMPLETE` |
| recommended IAC action | **DEFER** |

None of FIG-B1–B6 are `CURRENT_CODE_FAITHFUL`. Cropping the January PDF and labeling it as current ARTPS is **forbidden**.

### 2.2 Quantitative figures / tables

#### FIG-B7 / TAB-B2 — Historical grouped AUROC/AUPRC/F1 + anomaly comparison

| Field | Value |
|-------|--------|
| asset_id | FIG-B7 / TAB-B2 |
| PDF page | 22 (chart + Table 2) |
| historical caption | Grouped bar chart of Table 2 (AUROC/AUPRC/F1). Table: Anomaly detection performance comparison. |
| scientific purpose | State B detection headline (accepted-IAC subset lives here) |
| historical values | ARTPS AUROC **0.894** / AUPRC **0.847** / F1 **0.823** / FPR 0.089; OpenCV 0.723 / 0.645 / 0.612 / 0.234; Depth-only 0.781 / 0.698 / 0.689 / 0.187; PaDiM/PatchCore 0.856 / 0.812 / 0.794 / 0.134 |
| generation script located? | `8f7e3ff:scripts/generate_paper_charts.py` hard-codes ARTPS 0.894/0.847/0.823 and extra baseline AUPRC/F1 |
| prediction/run artifact located? | **no** (archaeology: visualization constant, not raw experiment) |
| current-code equivalent exists? | IAC Table 1 already reports accepted 0.894/0.847/0.823/0.856/28.1 as **historical pending**; extra AUPRC/F1/FPR **not** in IAC |
| claim layer | C05/C06 accepted abstract (subset) + extra unpinned baselines |
| evidence status | Accepted subset: `HISTORICAL_REPORTED_ONLY` (already in IAC Table 1). Extra baselines / FPR / Fig 7: `UNSUPPORTED_FOR_CURRENT_IAC` without run provenance |
| recommended IAC action | **KEEP** accepted subset in Table 1 only. **Do not copy Fig 7** or extra columns |

#### TAB-B3 — Historical depth metrics

| Field | Value |
|-------|--------|
| asset_id | TAB-B3 |
| PDF page | 23 |
| historical caption | Depth estimation performance comparison |
| historical values | RAE 0.156 vs 0.234; RMSE 0.189 vs 0.287; MAE 0.134 vs 0.198; Log10 0.089 vs 0.145; δ<1.25 89.4% vs 76.8%; δ<1.25² 97.8% vs 89.2%; δ<1.25³ 99.2% vs 95.7% |
| GT source / scale / alignment / predictions | **not located** |
| current-code equivalent exists? | monocular **relative** depth only; metric RAE/RMSE **forbidden** (C15) |
| evidence status | `HISTORICAL_REPORTED_ONLY` / `UNSUPPORTED_FOR_CURRENT_IAC` |
| recommended IAC action | **DO NOT IMPORT** |

#### TAB-B4 — Historical curiosity ranking

| Field | Value |
|-------|--------|
| asset_id | TAB-B4 |
| PDF page | 23 |
| historical caption | Curiosity-score ranking performance comparison |
| historical values | nDCG@5/10/20 = 0.945 / 0.912 / 0.878 vs 0.712 / 0.734 / 0.689; Spearman 0.847 vs 0.623; Kendall τ 0.689 vs 0.456 |
| relevance labels / ranking run artifacts | **not located** |
| current-code equivalent exists? | UI ranking `implemented` (C03); v1_1 frozen score **excludes** curiosity / diversity / buffer |
| evidence status | `HISTORICAL_REPORTED_ONLY` / `UNSUPPORTED_FOR_CURRENT_IAC` |
| recommended IAC action | **DO NOT IMPORT**; cannot attach to v1_1 |

#### FIG-B8 — Historical ablation chart

| Field | Value |
|-------|--------|
| asset_id | FIG-B8 |
| PDF page | 24 |
| historical caption | Ablation impact (AUROC drops vs nDCG drop for curiosity) |
| historical values | depth −9.2% (0.894→0.812); enhancement −4.2% (0.894→0.856); fusion −16.9% (0.894→0.743); curiosity nDCG −25.7% (0.912→0.678) |
| generation script located? | same hard-coded chart script |
| experiment artifacts / configs | **not located** |
| evidence status | `HISTORICAL_REPORTED_ONLY` / `UNSUPPORTED_FOR_CURRENT_IAC` |
| recommended IAC action | **DO NOT IMPORT**; regenerating ablations now would be a **new experiment** |

#### FIELD-B — Field-condition AUROC/FPR

| Field | Value |
|-------|--------|
| asset_id | FIELD-B |
| PDF page | 24 |
| historical values | low texture 0.867 / 0.112; high contrast 0.912 / 0.067; shadow-dense 0.843 / 0.134; far field 0.789 / 0.198 |
| eval sets / predictions | **not located** |
| current-code equivalent exists? | C09–C11 **proxy** only (non-headline); not these numbers |
| evidence status | `HISTORICAL_REPORTED_ONLY` |
| recommended IAC action | **DO NOT PROMOTE** |

#### FPS-B — Resolution / runtime

| Field | Value |
|-------|--------|
| asset_id | FPS-B |
| PDF page | 24 |
| historical values | **28.1 FPS** @ 256×256 (~35.6 ms); 12.8 FPS @ 384×384; 4.0 FPS @ 768×768; lightweight OpenCV core (no learned depth/AE) |
| script located? | historical `benchmark_cv_core_speed.py` @ `8f7e3ff` / HEAD surrogate `historical_opencv_surrogate_8f7e3ff` |
| pinned 28.1 timing JSON | **no** (C07 still pending) |
| evidence status | 28.1: `HISTORICAL_REPORTED_ONLY` (already IAC Table 1). 12.8 / 4.0: `HISTORICAL_REPORTED_ONLY` |
| recommended IAC action | **KEEP** 28.1 as accepted historical only. **Do not add** 12.8 / 4.0 unless C07 closes |

#### DHE-B / SAFETY-B

| asset_id | PDF / TeX | Values | Status | IAC action |
|----------|-----------|--------|--------|------------|
| DHE-B | §3.2.1 + FIG-B1 | qualitative “DHE off/on” mentioned; no isolated DHE metric table located | `PROVENANCE_INCOMPLETE`; C13 Real-ESRGAN detection gain `unsupported` | **DEFER** |
| SAFETY-B | prose FPR 0.089 vs 0.234; shadow/specular suppression | same unpinned FPR as TAB-B2 | `HISTORICAL_REPORTED_ONLY` | **DO NOT IMPORT** as measured reliability |

#### QUAL-EXTRA — `paper_figs/summary.csv` / `paper_report.md` overlays

| Field | Value |
|-------|--------|
| asset_id | QUAL-EXTRA |
| historical files | e.g. `0735MR0031500040403079E01_DXXX.jpg` + 4 other `paper_images` rows; detection overlays in `paper_report.md` |
| scientific purpose | exploratory curiosity / overlay dump |
| evidence status | `UNSUPPORTED_FOR_CURRENT_IAC` as Results; top-k curiosity = **cherry-pick** |
| recommended IAC action | **DO NOT USE** as sample-selection seed |

---

## 3. Counts

| Category | Count |
|----------|-------|
| Historical figures audited (B1–B8) | 8 |
| Extra qualitative dump (QUAL-EXTRA) | 1 group |
| Historical tables audited (architecture T1 + anomaly T2 + depth T3 + ranking T4) | 4 |
| Field / FPS / DHE / safety result blocks | 4 |
| Assets with **full** C05-style run provenance (predictions + split + SHA manifest + config) | **0** |
| Assets with **partial** provenance (git blob and/or hard-coded chart script and/or TeX numbers) | FIGS B1–B8, TAB-B2–B4, FPS-B, FIELD-B |
| Assets with **no** recoverable file blob | extra baseline run CSVs; depth GT; nDCG relevance labels; field-condition sets |
| Quantitative assets safe for current IAC **as already reported historical** | accepted 0.894 / 0.847 / 0.823 / 0.856 / 28.1 only |
| Quantitative assets deferred | extra AUPRC/F1/FPR, depth table, ranking table, Fig 7–8, field AUROC/FPR, extra FPS |
| Qualitative assets safe to reuse as **current-code** output | **0** |
| Qualitative assets regeneratable without a new experiment | **NO** (would require ARTPS inference + new overlays) |

---

## 4. Decision gate

Preferred content *if* a later PR ever adds one qualitative figure: RGB / recon-or-fused cue / relative depth / valid-candidate overlay; caption: qualitative illustration, not a performance estimate, relative depth non-metric, frozen image-score path vs ranking not implied. **Not this PR.**

### Recommended decision: **B** — KEEP_ARCHITECTURE_FIGURE_ONLY

| Test | Result |
|------|--------|
| A — current-code-faithful, non-test, deterministic regen, **not** a new experiment | **FAIL**. Regen requires inference → new experiment. No pinned config/checkpoint/split linking B1–B6 to current frozen eval. Informal filenames ≠ PDS manifest. |
| C — historical qualitative reuse, honest label, **clear scientific value** | **FAIL as required action**. Blobs exist at `e4282a3`, but PDS/process provenance is weak; Jan-2026 overlays ≠ current fusion/masks/score contract; IAC already has a code-faithful TikZ architecture figure. Risk of readers treating photos as current ARTPS outweighs incremental reviewer value. |
| B — architecture Fig.~1 only | **PASS**. Evidence-default when A and C fail. |

Confidence: **high**.

---

## 5. Author decision report (also in chat)

1. Recommended decision: **B**
2. Confidence: **high**
3. Provenance evidence: State B TeX+PDF inventory complete; qualitative blobs at `e4282a3:ARTPS_repo/`; Fig 7/8 hard-coded in `generate_paper_charts.py`; accepted metrics already in IAC Table 1; freeze closed; C05–C07 pending
4. Missing: PDS IDs / historical SHA manifest for B1–B6; C05 prediction CSVs; depth GT + scale protocol; nDCG labels; field-condition sets; ablation run configs; chart script not on HEAD
5. Candidate historical figures: FIG-B2 / FIG-B6 most illustrative **if** a future C were approved; **not recommended now**
6. Current-code regeneration possible? **NO** without new inference (and not this PR)
7. Test split required? **NO** (must remain unused)
8. Would regeneration constitute a new experiment? **YES**
9. Expected manuscript benefit of adding a figure now: low (TikZ Fig 1 already explains A/B/C layers)
10. Expected scientific/evidence risk of adding now: high (misread as current measured output; metric-depth / ranking / ablation leakage)
11. Recommended exact manuscript action: **no TeX change**; keep Fig.~1 only; do not import State B tables/charts; leave email TBD; no `SUBMISSION_FREEZE.md` here

`AWAITING_AUTHOR_REVIEW_BEFORE_TEX_OR_FIGURE_CHANGES`

---

## 6. Submission-freeze recommendation (after author review)

If author accepts **B**: current IAC MS is scientifically freeze-ready **except** `CORRESPONDING_EMAIL_TBD` (`AUTHOR_ACTION_REQUIRED`). Next PR = email + optional Word-template slot, not legacy figures.

If author later requests **A** or **C**: new PR after explicit approval; still no test inference; still no C05/C06/C07 closure by assumption.
