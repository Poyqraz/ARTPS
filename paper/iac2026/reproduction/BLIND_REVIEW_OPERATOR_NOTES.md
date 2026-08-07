# Blind validation re-review — operator notes (repeat-author scaffold)

## Status

- Pack built locally (gitignored): `results/iac2026/independent_eval_v1/blind_review_pack/`
- `ARTPS_DATASET_ROOT` = repo `mars_images/` (54/54 validation paths present)
- This pass is a **model-blind repeat visual review by the author**
- `review_type: repeat_author_blind_review`
- `independent_annotator: false`
- Do **not** describe this as an independent second annotation

## How to run (author)

```powershell
$env:ARTPS_DATASET_ROOT = "C:\Users\cancor\Desktop\Repos\project_mars\mars_images"
python scripts/iac2026/build_validation_blind_review_pack.py
streamlit run scripts/iac2026/annotate_validation_blind_review.py -- `
  --pack-dir results/iac2026/independent_eval_v1/blind_review_pack `
  --repeat-author-review
```

`--repeat-author-review` is required.

Shows: neutral `review_XXXX.jpg`, annotation guide, raw label buttons (`positive` / `negative` / `uncertain` / `exclude`), confidence, notes.

Does **not** show: sample_id, relative_path, split, terrain, binary_label, model scores, candidates.

On each save:
- raw label stays in `review_queue.csv` (`positive`/`negative`/…)
- canonical export refreshes `blind_review_results.csv` (`reviewer_label_raw` + normalized `0`/`1`/…)

## After all 54 reviewed

```powershell
python scripts/iac2026/compare_blind_review_labels.py
```

Fail-closed until 54 unique completed reviews. Comparison uses private mapping + original manifest labels only (no model scores). Does not mutate the manifest.

## Do not commit

Pack images, `private_mapping.csv`, filled `review_queue.csv` / `blind_review_results.csv`, or `blind_review_analysis/`.

## Freeze

Final test remains closed (`blocked_validation_sanity_review`). This review does not open the test split.
