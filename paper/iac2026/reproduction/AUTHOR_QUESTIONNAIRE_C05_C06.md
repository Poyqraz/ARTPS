# Author questionnaire — C05 / C06 (evidence blockers only)

Generated after deep salt-okunur archaeology on branch `experiments/iac2026-c05-c06-dataset`.
Only fields that remain **not** evidence-closed are listed.

**Allowed answers include:** exact artifact paths, quotes, “hatırlamıyorum”, “dosya yok / silindi”.
Do **not** invent values to unblock a run.

Evidence summary: [C05_C06_DEFINITION_AUDIT.md](C05_C06_DEFINITION_AUDIT.md),
[DATASET_MANIFEST_GAPS.md](DATASET_MANIFEST_GAPS.md).

**Recorded status (audit observation — not author attestation):**
[C05_C06_RESPONSE_STATUS.md](C05_C06_RESPONSE_STATUS.md).

**Author attestation (pending template — do not invent fills):**
[AUTHOR_ATTESTATION_C05_C06.template.md](AUTHOR_ATTESTATION_C05_C06.template.md).

---

## P0 — blocks any real C05/C06 reproduction run

### Q-P0-01 — Evaluation unit (task level)

**Question:** Accepted table metrics (AUROC / AUPRC / F1) were computed at which unit — image-level binary scores, pixel-level maps, or region-level scores?

**Why required:** HEAD real_evidence harness is image_binary-only; wrong unit invalidates the runner contract.

**What was found:** Manuscript evaluates “anomaly discrimination (AUROC/AUPRC/F1/FPR)” without stating the sample unit (`8f7e3ff:docs/paper.tex` abstract; Full PDF metrics section). No executable definition recovered.

**What exact artifact/answer would close it:** One of: (a) prediction CSV with one score per image + statement “image-level”; (b) notebook/script computing the published metrics with explicit aggregation; (c) written protocol excerpt naming the unit.

**Acceptable response format:**
- Free text + optional file path
- Or: `image-level` / `pixel-level` / `region-level` / `hatırlamıyorum` / `bilmiyorum`
- If choosing a level, cite source. Candidates **with sources** (not suggestions to pick casually):
  - *None located as executable.* Manuscript language is compatible with multiple units — do not guess.

---

### Q-P0-02 — Positive / negative class definition

**Question:** What was the positive class for the published F1/AUROC/AUPRC (operational label rule)?

**Why required:** Labels and `positive_label` must match the historical experiment.

**What was found:** No annotation dictionary, GT CSV, or label_semantics for the 2,847-image claim set. Synth fixture uses `positive_label: 1` / `anomaly_binary` (software verification only).

**What exact artifact/answer would close it:** Label guide / codebook / labeling notebook + version; or a sentence with examples of positive vs negative images used in the table.

**Acceptable response format:** Text definition + optional artifact path; or `hatırlamıyorum` / `etiket dosyası yok`.

---

### Q-P0-03 — Per-file train / validation / test membership

**Question:** Does a split file (CSV/JSON/list) still exist that maps each image to train/validation/test for the published table?

**Why required:** Aggregate counts (2847 / 1247 / 892 / 708) cannot legally become a manifest.

**What was found:** Manuscript counts + protocol “identical train/validation/test splits” (`8f7e3ff:docs/paper.tex`). **No** per-file membership in git history. 1247+892+708=2847 arithmetic observed; whether that is a disjoint partition remains UNKNOWN ([DATASET_MANIFEST_GAPS.md](DATASET_MANIFEST_GAPS.md)).

**What exact artifact/answer would close it:** Split CSV with `sample_id`/`path`/`split` (and ideally sha256); or archive link.

**Acceptable response format:** Path/URL to split file; or `yok` / `hatırlamıyorum`. Optional clarification: are 1247 / 892 / 708 disjoint buckets or mission totals overlapping the 708 pool?

---

### Q-P0-04 — F1 threshold selection details

**Question:** How was the decision threshold for the published F1=0.823 chosen on validation (metric, tie-break, numeric threshold if retained)?

**Why required:** F1 needs a threshold; manuscript only says validation-selected.

**What was found:** “Decision thresholds are selected on validation data; results are reported on the test set.” (`8f7e3ff:docs/paper.tex` Protocol). Numeric value / selection metric / tie-break **not located**.

**What exact artifact/answer would close it:** Validation selection script output, logged threshold float, or notebook cell; plus metric name (e.g. F1) and tie-break if any.

**Acceptable response format:** `{metric, tie_break, threshold}` JSON-like; or path; or `hatırlamıyorum`.

---

### Q-P0-05 — Meaning of published AUPRC 0.847

**Question:** Was 0.847 computed as sklearn-style average precision, trapezoidal PR-AUC, or another PR integral?

**Why required:** HEAD harness distinguishes `average_precision` vs `trapezoidal_pr_auc`; accepted claim must name one.

**What was found:** Manuscript/abstract label the number **AUPRC** / “area under the Precision–Recall curve” (Full PDF §4.3.1). Method discrimination **not located**. Chart scripts hard-code 0.847 (visualization constant only).

**What exact artifact/answer would close it:** Metrics code path that produced the table, or explicit method name in a lab note.

**Acceptable response format:** `average_precision` / `trapezoidal_pr_auc` / `other:<name>` / `hatırlamıyorum` — only if you have a source; otherwise say unknown.

---

### Q-P0-06 — Identity of baseline AUROC 0.856

**Question:** Does 0.856 belong to PaDiM alone, PatchCore alone, the better of the two, an average, or another combined rule?

**Why required:** Single manuscript column “PaDiM/PatchCore (WRN-50-2)” is ambiguous; HEAD stubs explicitly refuse inventing an average.

**What was found:** Table header `PaDiM/PatchCore (WRN-50-2)` with AUROC 0.856 (`8f7e3ff:docs/paper.tex`; Full PDF Table 2; accepted abstract). Backbone name WRN-50-2 located as manuscript claim. Separate PaDiM vs PatchCore scores **not located**. Local `results/padim_stats.pth` / `results/patchcore_bank.pth` exist as **PATH_REFERENCE_ONLY** (provenance/SHA unverified).

**What exact artifact/answer would close it:** Baseline prediction CSV; or lab note naming which system produced 0.856; or separate PaDiM and PatchCore AUROCs from the same split.

**Acceptable response format:** One of `padim_only` / `patchcore_only` / `best_of_two` / `mean_of_two` / `other:<desc>` **with source**; or `hatırlamıyorum`. Do not pick mean_of_two without evidence.

---

### Q-P0-07 — Raw predictions or re-inference recipe

**Question:** Do pinned prediction tables (ARTPS + baseline) still exist, or is there a complete re-inference recipe (checkpoint SHA, config, split, preprocessing) that regenerates them?

**Why required:** Without predictions or a closed recipe, metrics cannot be recomputed under audit.

**What was found:** No C05/C06 prediction CSV in git. Synth fixture only. Weight paths present locally without verified SHA/recipe link to 0.856.

**What exact artifact/answer would close it:** `predictions.csv` matching harness schema; or documented recipe + weight SHAs + split.

**Acceptable response format:** Paths; or `yok` / `hatırlamıyorum`.

---

## P1 — reproducibility quality (run still blocked without P0)

### Q-P1-01 — Random seed value(s)

**Question:** What numeric seed(s) were used for the published detection table?

**Why required:** Protocol claims “consistent random seeds” without values.

**What was found:** Protocol sentence only (`8f7e3ff:docs/paper.tex`).

**What exact artifact/answer would close it:** Seed integer(s) in config/log/notebook.

**Acceptable response format:** integer list; or `hatırlamıyorum`.

---

### Q-P1-02 — Eval preprocessing / input resolution

**Question:** Exact resize/crop/normalize used when scoring the published metrics?

**Why required:** Manuscript states resolution span and CLAHE/gamma narrative only.

**What was found:** Dataset preprocessing bullets; metric input size UNKNOWN.

**What exact artifact/answer would close it:** Config YAML / script args / notebook.

**Acceptable response format:** short recipe + path; or `hatırlamıyorum`.

---

### Q-P1-03 — PaDiM / PatchCore bank recipes

**Question:** Which layers, image size, coreset (PatchCore), and train-split membership produced the local `.pth` banks?

**Why required:** PATH_REFERENCE files alone cannot verify 0.856.

**What was found:** Modules/prepare scripts exist @ `8f7e3ff` (`tools/prepare_padim_stats.py`, `tools/prepare_patchcore_bank.py`); bank SHA/provenance not in git.

**What exact artifact/answer would close it:** Prepare command logs + `sha256` of weight files + train ID list.

**Acceptable response format:** paths + hashes; or `hatırlamıyorum`.

---

### Q-P1-04 — Scene / duplicate leakage controls

**Question:** Were same-Sol / same-scene / duplicate-hash images prevented from crossing splits?

**Why required:** Leakage can inflate AUROC/AUPRC/F1.

**What was found:** No scene_group / duplicate policy recovered.

**What exact artifact/answer would close it:** Policy note + grouping columns in the split file.

**Acceptable response format:** description + file; or `uygulanmadı` / `hatırlamıyorum`.

---

## P2 — manuscript clarity (not required to start engineering once P0 closed)

### Q-P2-01 — Shared-split claim strength

**Question:** Can you confirm that ARTPS and the PaDiM/PatchCore column used the **same** test IDs (beyond the protocol sentence)?

**Why required:** Hard rule: same table ≠ automatic shared split proof.

**What was found:** Protocol claims identical splits; no ID intersection proof.

**What exact artifact/answer would close it:** Shared `sample_id` lists or a single split file used by both.

**Acceptable response format:** confirmation + path; or `emin değilim` / `hatırlamıyorum`.

---

### Q-P2-02 — Ambiguous Spearman 0.847

**Question:** Note only: ranking tables also report Spearman **0.847**. Confirm detection AUPRC 0.847 is distinct from that ranking metric in your records.

**Why required:** Avoid cross-metric confusion in future prose (not a run blocker by itself).

**What was found:** Same numeric token appears in ranking tables (`8f7e3ff:docs/paper.tex`).

**What exact artifact/answer would close it:** Separate logs for detection vs ranking.

**Acceptable response format:** short confirmation; or `hatırlamıyorum`.
