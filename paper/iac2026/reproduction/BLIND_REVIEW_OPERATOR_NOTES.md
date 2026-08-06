# Blind validation re-review — operator notes (post PR #26 merge)

## Status

- Pack built locally (gitignored): `results/iac2026/independent_eval_v1/blind_review_pack/`
- `ARTPS_DATASET_ROOT` = repo `mars_images/` (54/54 validation paths present)
- `independent_review_status`: **pending** (no second independent annotator yet)
- Author may run a **repeat** pass with `--repeat-author-review` (does not clear the pending independent requirement)

## How to run

```powershell
$env:ARTPS_DATASET_ROOT = "C:\Users\cancor\Desktop\Repos\project_mars\mars_images"
streamlit run scripts/iac2026/annotate_validation_blind_review.py -- `
  --pack-dir results/iac2026/independent_eval_v1/blind_review_pack `
  --repeat-author-review
```

Shows: neutral `review_XXXX.jpg`, annotation guide, label/confidence/notes only.
Does **not** show: sample_id, relative_path, split, terrain, binary_label, model scores.

Do **not** commit pack images, `private_mapping.csv`, or filled reviewer fields.

## Freeze

Final test remains closed (`blocked_validation_sanity_review`). This review does not open the test split.

<!-- ci: retrigger PR27 checks -->

