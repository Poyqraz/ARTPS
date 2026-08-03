# Dataset manifest gaps (C05 / C06)

Evidence-only. Aggregate manuscript counts must **not** be expanded into fabricated
per-file rows.

Template (headers only): [`reproduction/iac2026/manifests/c05_c06_manifest.template.csv`](../../../reproduction/iac2026/manifests/c05_c06_manifest.template.csv).

---

## 1. Aggregate information known from evidence

| Claim | Value | Source | Tag |
|-------|-------|--------|-----|
| Total images (stated) | 2,847 | Full PDF §4.1; `8f7e3ff:docs/paper.tex` Dataset | MANUSCRIPT_CLAIM |
| Curiosity Mastcam | 1,247 (Sol 100–1700) | Same | MANUSCRIPT_CLAIM |
| Perseverance Mastcam-Z | 892 (Sol 1–400) | Same | MANUSCRIPT_CLAIM |
| Test/validation bullet | 708 “under diverse field conditions” | Same | MANUSCRIPT_CLAIM |
| Arithmetic | 1247 + 892 + 708 = 2847 | Observation | INFERENCE_NOT_ALLOWED as proof of disjoint partition |
| Protocol | “identical train/validation/test splits”, “consistent random seeds”, validation-selected thresholds, test reporting | `8f7e3ff:docs/paper.tex` Protocol | MANUSCRIPT_CLAIM |
| Preprocess narrative | resolution equalization, denoising, CLAHE, gamma | Dataset section | MANUSCRIPT_CLAIM |
| Resolution span | 640×480 to 1920×1080 | Dataset section | MANUSCRIPT_CLAIM |

**Ambiguity (explicit):** whether 1,247 / 892 are mission **source** counts that **include** the 708 test/val images, or three **disjoint** partition buckets, is **UNKNOWN_EVIDENCE_NOT_LOCATED**. Manuscript wording (“split as follows”) is insufficient without a per-file list.

---

## 2. Per-file information known

| Item | Status |
|------|--------|
| SHA256-pinned NASA image list for C05/C06 | **Not located** |
| `sample_id` ↔ split membership | **Not located** |
| `binary_label` / `label_semantics` per image | **Not located** |
| `scene_group_id` / `duplicate_group_id` | **Not located** |
| `source_url` / PDS product id | **Not located** |
| Synthetic SW fixture rows | Present under `reproduction/iac2026/fixtures/synthetic_manifest.csv` — **not** the claim dataset |

`results/paper_figs/summary.csv` @ `8f7e3ff` lists a few `results/paper_images/...` paths with curiosity scores — **not** a labeled C05 detection split.

---

## 3. Gaps required to build a real manifest

Must be evidence-backed before a non-template CSV is accepted:

1. Stable `sample_id` scheme
2. `mission`, `instrument`, `sol` (or explicit null policy)
3. `source_id` (PDS / product id) and preferably `source_url`
4. `relative_path` under a documented `dataset_root_env`
5. `sha256` of file bytes
6. `split` ∈ {train, validation, test} with **no** unassigned claim rows
7. `binary_label` + `label_semantics` + `label_source` + `annotation_version`
8. Optional but strongly recommended: `scene_group_id`, `duplicate_group_id`
9. Provenance note: who labeled, when, tool/version

Without per-file `source_id` + `sha256`, the manifest is **not ready**.

---

## 4. Leakage-control fields

| Risk | Required field / check | Status |
|------|------------------------|--------|
| Same Sol/scene in train and test | `scene_group_id` (+ policy) | UNKNOWN |
| Duplicate / re-encoded / renamed files | `duplicate_group_id` + sha256 | UNKNOWN |
| Multi-name same image in local trees | Content hash collision audit | Not run (no pinned set) |
| Train contamination of baseline bank | Document which split built PaDiM/PatchCore banks | UNKNOWN |

---

## 5. Author must provide (cannot be invented here)

- Per-file split CSV or equivalent export from the original experiment
- Random seed(s) used for the published table
- Label dictionary / positive-class definition
- Annotation files or labeling notebook with version pin
- Prediction table that produced 0.894 / 0.847 / 0.823 (if retained)
- Baseline prediction table or exact recipe for 0.856
- Confirmation whether 1247/892/708 are disjoint partitions

Acceptable: “hatırlamıyorum” / “dosya yok” — then fields stay UNKNOWN.

---

## 6. Re-downloadable imagery vs original annotations

| Asset | Re-downloadable? | Notes |
|-------|------------------|-------|
| NASA PDS Mastcam / Mastcam-Z rasters | Often yes (public) | Re-download ≠ reconstruct labels/splits/SHA of the original run |
| Human / semi-auto binary labels | **No** (unless author archives) | Original annotations are unique evidence |
| Train banks `padim_stats.pth` / `patchcore_bank.pth` | Local PATH_REFERENCE_ONLY | Rebuilding without recipe ≠ verifying 0.856 |
| Accepted abstract numbers | Manuscript claim only | Not a substitute for predictions |

**Do not** generate rows by sampling local `mars_images/` to “match” 2847.
