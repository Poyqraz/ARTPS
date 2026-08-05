# Independent evaluation v1 — primary domain selection

**Protocol:** [INDEPENDENT_EVALUATION_PROTOCOL.md](INDEPENDENT_EVALUATION_PROTOCOL.md)

**Selection rule:** choose one narrow primary domain using inventory stats only.
Model scores, ARTPS outputs, and heatmaps were **not** used.

**Dataset root:** `ARTPS_DATASET_ROOT` → workspace `mars_images/` (read-only inventory).

## Inventory snapshot (source_inventory)

| Metric | Value |
|--------|------:|
| Files | 2584 |
| Readable | 2584 |
| Exact-SHA duplicate members | 0 |
| Mission Curiosity (filename token) | 1755 |
| Mission UNKNOWN | 829 |
| Instrument Mastcam (filename token) | 1286 |
| Instrument UNKNOWN | 1298 |

Folder counts under `train/` + `valid/` (path taxonomy only; **not** binary GT):
boulder 493, rocky 525, hills_or_ridge 446, flat_terrain 488, dusty 346, rover 274, Unlabeled 3; plus 9 root orphans.

## Candidate domains

| Candidate domain | Available files | Readable | Metadata completeness | Duplicate risk | Annotation suitability | Decision |
|---|---:|---:|---:|---:|---:|---|
| Curiosity Mastcam (filename `curiosity` + `MAST` / `mastcam`) under `train/`+`valid/` | ~1200+ | all readable in inventory | partial (sol/product often UNKNOWN; mission+instrument often known) | low (0 exact SHA dups in full root) | high — homogeneous 640×640 Roboflow Mastcam-style RGB | **PRIMARY** |
| Curiosity + any instrument (incl. FHAZ/other) under `train/`+`valid/` | larger | all readable | weaker instrument homogeneity | low | medium — mixes camera geometries | reject for primary |
| Path folder `rover` only | 274 | readable | partial | low | poor — hardware-dominated; guide §C | exclude from primary (stress/future) |
| Root orphans / Unlabeled | 12 | readable | none | low | poor | exclude from primary |
| Perseverance / Mastcam-Z (local) | 0 in this root | — | — | — | — | not available; future stress set |

## Locked primary domain

**ID:** `curiosity_mastcam_roboflow_v1`

**Inclusion for annotation queue:**

- `relative_path` under `train/` or `valid/`
- path folder ≠ `rover` and ≠ `Unlabeled`
- filename tokens indicate Curiosity Mastcam (`curiosity` and `mast` / `MAST` / `mastcam`) when present; if mission/instrument parse is UNKNOWN but path is otherwise in the homogeneous Roboflow Mastcam-sized corpus **and** filename contains `curiosity` + `MAST`, include
- readable = true

**Explicit non-mix:** do not add Perseverance, FHAZ-only products, or rover-folder frames into the primary table.

## Honest limits

- Terrain folder names are **not** `binary_label` ground truth.
- Labels use `label_source=workspace_visual_review`, `annotator_count=1`, `independent_double_review=false`.
- No recovered historical C05 annotation archive.
