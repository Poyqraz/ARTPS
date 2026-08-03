# C05 / C06 definition audit

Salt-okunur archaeology (branch `experiments/iac2026-c05-c06-dataset`).
No inference, no fabricated splits, no claim promotion.

**Scan scope:** `main` / all remotes / tags / full history (`git log -S` / `-G`),
baseline `8f7e3ff`, stash@{0} name listing only (not applied), local
`docs/*.pdf`, `results/*.pth` presence, HEAD harness under `reproduction/iac2026/`.

**Source-type tags:** `RAW_ARTIFACT` · `EXECUTABLE_DEFINITION` · `MANUSCRIPT_CLAIM` ·
`VISUALIZATION_CONSTANT` · `PATH_REFERENCE_ONLY` · `INFERENCE_NOT_ALLOWED` ·
`UNKNOWN_EVIDENCE_NOT_LOCATED`

**Row status:** `LOCATED` · `PARTIALLY_LOCATED` · `UNKNOWN_EVIDENCE_NOT_LOCATED`

Hard rule: co-appearance in the same manuscript table does **not** by itself prove
identical dataset/split membership for C05 vs C06. A separate protocol sentence claiming
identical splits is recorded as a manuscript claim, not as a recovered file list.

---

## Field audit table

| Field | C05 evidence | C06 evidence | Source type | Exact source/commit/page | Confidence | Status | Notes |
|------|--------------|--------------|-------------|--------------------------|------------|--------|-------|
| task_level | Narrative “anomaly discrimination (AUROC/AUPRC/F1/FPR)” without image/pixel/region unit | Same table metrics; no spatial unit stated | MANUSCRIPT_CLAIM | `8f7e3ff:docs/paper.tex` abstract; Full PDF §metrics | low | UNKNOWN_EVIDENCE_NOT_LOCATED | HEAD schema allows `image_binary`/`pixel_binary`/`region_binary`; historical value not recovered |
| positive_class | Not defined operationally | Not defined | UNKNOWN_EVIDENCE_NOT_LOCATED | — | none | UNKNOWN_EVIDENCE_NOT_LOCATED | Synth fixture uses `positive_label: 1` / `anomaly_binary` — SW only |
| negative_class | Not defined | Not defined | UNKNOWN_EVIDENCE_NOT_LOCATED | — | none | UNKNOWN_EVIDENCE_NOT_LOCATED | — |
| anomaly_score_definition | “anomaly discrimination”; fusion/maps described; image-level score aggregation not pinned | PaDiM Mahalanobis map formula in MS; image aggregation UNKNOWN | MANUSCRIPT_CLAIM | `8f7e3ff:docs/paper.tex` PaDiM formula; fusion section | low | UNKNOWN_EVIDENCE_NOT_LOCATED | Executable map code exists for viz; not a pinned C05 score recipe |
| score_orientation | Not pinned | Not pinned | UNKNOWN_EVIDENCE_NOT_LOCATED | — | none | UNKNOWN_EVIDENCE_NOT_LOCATED | Example configs assume higher=more anomalous (template only) |
| sample_unit | Metrics presented as AUROC/AUPRC/F1 without stating image vs pixel samples | Same | MANUSCRIPT_CLAIM | Table `tab:anomaly-performance` @ `8f7e3ff:docs/paper.tex` | low | UNKNOWN_EVIDENCE_NOT_LOCATED | — |
| dataset_sources | NASA PDS; Curiosity Mastcam; Perseverance Mastcam-Z | Same narrative table/protocol | MANUSCRIPT_CLAIM | Full PDF p.18 §4.1; `8f7e3ff:docs/paper.tex` Dataset section | medium | PARTIALLY_LOCATED | Missions/instruments named; per-file source IDs absent |
| exact_sample_count | Claims 2,847 total; 1,247 Curiosity; 892 Perseverance; 708 test/validation | Same manuscript counts | MANUSCRIPT_CLAIM | Full PDF p.18; `paper_en.tex` @ `8f7e3ff` L245 | medium | PARTIALLY_LOCATED | 1247+892+708=2847 arithmetic identity; **source-count vs split-count membership UNKNOWN** |
| train_split | Protocol claims train/val/test exist | Protocol claims identical splits | MANUSCRIPT_CLAIM | `8f7e3ff:docs/paper.tex` Protocol paragraph | low | UNKNOWN_EVIDENCE_NOT_LOCATED | No per-file train list / seed |
| validation_split | Mentioned; used for threshold selection narrative | Same | MANUSCRIPT_CLAIM | Protocol: “Decision thresholds are selected on validation data” | low | UNKNOWN_EVIDENCE_NOT_LOCATED | Membership UNKNOWN |
| test_split | “results reported on the test set”; 708 “Test/validation” bullet | Same | MANUSCRIPT_CLAIM | Dataset bullets + Protocol | low | UNKNOWN_EVIDENCE_NOT_LOCATED | Ambiguous whether 708 is test+val combined or a pool |
| scene_grouping | Not found | Not found | UNKNOWN_EVIDENCE_NOT_LOCATED | — | none | UNKNOWN_EVIDENCE_NOT_LOCATED | Required for leakage control |
| duplicate_policy | Not found | Not found | UNKNOWN_EVIDENCE_NOT_LOCATED | — | none | UNKNOWN_EVIDENCE_NOT_LOCATED | Alias filenames likely in local trees |
| annotation_source | Not found | Not found | UNKNOWN_EVIDENCE_NOT_LOCATED | — | none | UNKNOWN_EVIDENCE_NOT_LOCATED | No GT CSV recovered |
| label_semantics | Not found (beyond “anomaly”) | Not found | UNKNOWN_EVIDENCE_NOT_LOCATED | — | none | UNKNOWN_EVIDENCE_NOT_LOCATED | — |
| threshold_policy | Validation-selected decision thresholds (narrative) | Same protocol sentence | MANUSCRIPT_CLAIM | `8f7e3ff:docs/paper.tex` Protocol | medium | PARTIALLY_LOCATED | Policy class known as validation-selected; metric/tie-break/value UNKNOWN |
| fixed_threshold | Not stated | Not stated | UNKNOWN_EVIDENCE_NOT_LOCATED | — | none | UNKNOWN_EVIDENCE_NOT_LOCATED | — |
| threshold_selection_split | “validation data” | Same | MANUSCRIPT_CLAIM | Protocol paragraph | medium | PARTIALLY_LOCATED | Split name only; no file list |
| PR metric definition | Labeled **AUPRC**; “area under the Precision–Recall curve” | Baseline AUPRC 0.812 in same table | MANUSCRIPT_CLAIM | Full PDF §4.3.1; Table 2 | medium | PARTIALLY_LOCATED | Does **not** distinguish average_precision vs trapezoidal_pr_auc |
| random_seed | Protocol claims “consistent random seeds” | Same | MANUSCRIPT_CLAIM | Protocol paragraph | low | UNKNOWN_EVIDENCE_NOT_LOCATED | Numeric seed not located |
| preprocessing | Resolution equalization, denoising, CLAHE, gamma | “same preprocessing where applicable” | MANUSCRIPT_CLAIM | Dataset + Protocol @ `8f7e3ff` | medium | PARTIALLY_LOCATED | Not a pinned executable recipe for the published numbers |
| input_resolution | Images 640×480–1920×1080 span stated | Backbone size UNKNOWN | MANUSCRIPT_CLAIM | Dataset section | low | UNKNOWN_EVIDENCE_NOT_LOCATED | Eval crop/resize for metrics UNKNOWN |
| model/checkpoint | ARTPS hybrid stack (narrative) | Expected `results/padim_stats.pth`, `results/patchcore_bank.pth` | PATH_REFERENCE_ONLY + MANUSCRIPT_CLAIM | Local files present (~9.0 MB / ~53.0 MB, mtime 2025-08); code refs @ HEAD/`8f7e3ff` | low | PARTIALLY_LOCATED | Presence ≠ provenance; SHA/recipe/link to 0.856 UNKNOWN |
| C06 baseline identity | N/A | Single column “PaDiM/PatchCore (WRN-50-2)” | MANUSCRIPT_CLAIM | Table header @ `8f7e3ff:docs/paper.tex`; abstract | low | UNKNOWN_EVIDENCE_NOT_LOCATED | PaDiM-only / PatchCore-only / average / best not recoverable; HEAD stubs forbid fake average |
| PaDiM configuration | N/A | Backbone name WRN-50-2; Mahalanobis formula | MANUSCRIPT_CLAIM + EXECUTABLE_DEFINITION (module) | MS + `src/models/anomaly/padim.py` / `tools/prepare_padim_stats.py` @ `8f7e3ff` | low | PARTIALLY_LOCATED | layers / image size / train bank / SHA UNKNOWN |
| PatchCore configuration | N/A | Same combined column; WRN-50-2 | MANUSCRIPT_CLAIM + EXECUTABLE_DEFINITION | `patchcore.py` / `prepare_patchcore_bank.py` @ `8f7e3ff` | low | PARTIALLY_LOCATED | coreset / layers / SHA UNKNOWN |
| baseline dataset linkage | N/A | Protocol claims identical splits to ARTPS | MANUSCRIPT_CLAIM | Protocol paragraph | low | PARTIALLY_LOCATED | Claim only; files not located — do not treat as proven membership |
| raw predictions availability | Not located for C05 numbers | Not located for 0.856 | UNKNOWN_EVIDENCE_NOT_LOCATED | Synth only: `reproduction/iac2026/fixtures/synthetic_predictions.csv` | none | UNKNOWN_EVIDENCE_NOT_LOCATED | `results/paper_figs/summary.csv` is curiosity ranking, not C05 labels |
| reported AUROC 0.894 | Abstract + Table | — | MANUSCRIPT_CLAIM + VISUALIZATION_CONSTANT | `paper/iac2026/main.tex`; Full PDF Table 2; charts hardcode @ stash/`8f7e3ff` scripts | high (as claim) | LOCATED | **Claim only** — not a raw experiment artifact |
| reported AUPRC 0.847 | Abstract + Table | Baseline AUPRC 0.812 co-listed | MANUSCRIPT_CLAIM + VISUALIZATION_CONSTANT | Same | high (as claim) | LOCATED | Method AP vs trapezoid still UNKNOWN |
| reported F1 0.823 | Abstract + Table | Baseline F1 0.794 | MANUSCRIPT_CLAIM + VISUALIZATION_CONSTANT | Same | high (as claim) | LOCATED | Threshold value UNKNOWN |
| reported baseline AUROC 0.856 | — | Abstract + Table | MANUSCRIPT_CLAIM + VISUALIZATION_CONSTANT | Same | high (as claim) | LOCATED | Identity UNKNOWN (see C06 baseline identity) |

---

## Notable non-evidence (do not upgrade)

| Artifact | Why not closure evidence | Tag |
|----------|--------------------------|-----|
| `scripts/generate_paper_charts.py` hard-coded `[0.894, 0.847, 0.823]` / `[0.856, …]` | Chart constant | VISUALIZATION_CONSTANT |
| Proxy evals `run_iac_shadow_proxy_eval.py`, size-distance | Explicitly not C05/C06 | INFERENCE_NOT_ALLOWED if used as substitute |
| App fusion loading PaDiM/PatchCore | Visualization path | PATH_REFERENCE_ONLY / EXECUTABLE_DEFINITION (UI) |
| Synth fixtures under `reproduction/iac2026/fixtures/` | Software verification only | RAW_ARTIFACT (SW) |
| Local `mars_images/` trees | Unpinned; may contain aliases; no GT link | PATH_REFERENCE_ONLY |

---

## Scan commands executed (representative)

- `git log --all -S"0.894|0.847|0.856|0.823|2847|1247|892|708"`
- `git log --all -G"roc_auc|average_precision|PatchCore|PaDiM"`
- `git grep` / `git show 8f7e3ff:docs/paper.tex` (dataset, protocol, Table)
- PDF text extract: `docs/Full_Baydemir_ARTPS.pdf`, portal/T3 abstracts
- `git stash show --name-only -u stash@{0}` (list only)
- File size check: `results/padim_stats.pth`, `results/patchcore_bank.pth`

Cross-link: [ARCHAEOLOGY_REPORT.md](ARCHAEOLOGY_REPORT.md), [DATASET_MANIFEST_GAPS.md](DATASET_MANIFEST_GAPS.md), [AUTHOR_QUESTIONNAIRE_C05_C06.md](AUTHOR_QUESTIONNAIRE_C05_C06.md), [`reproduction/iac2026/C05_C06_DEFINITIONS.yaml`](../../reproduction/iac2026/C05_C06_DEFINITIONS.yaml).
