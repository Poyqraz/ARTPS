# Qualitative Fig. 2 provenance

Illustration only. Not a quantitative experiment and not a Results metric.

| Field | Value |
| --- | --- |
| 1. Source image ID | `ie1_train_boulder_curiosity_100_MAST_106_jpg.rf.78c4b8fcf62f2b932ebccf43bd427980.jpg_4dd2d929c0ee` |
| Manifest product_id | `curiosity_100_mast_106` |
| Relative path | `train/boulder/curiosity_100_MAST_106_jpg.rf.78c4b8fcf62f2b932ebccf43bd427980.jpg` |
| File SHA256 | `4dd2d929c0ee66f91b17c3f73b8bf127f23b6f6775511da3a735e927fbf2291a` |
| 2. Split | `train` (not test) |
| 3. Selection rule | included rows from `independent_eval_v1.csv`; drop `split==test` or `relative_path` under `test/`; sort by `sample_id` UTF-8 lexicographic; first existing file under `ARTPS_DATASET_ROOT`; **no score lookup before selection** |
| 4. Repository commit at generation | `e1607508375ac841b68252e548c7adf7440e81bd` (`main` @ PR #39); panel (b) relabel only |
| 5. Checkpoints | AE `results/optimized_autoencoder_curiosity_extended.pth` `8186bbe6be424dd212d5d4a93b1ae36b80939552519706b4a8680c5d05e995f2`; DPT_Large `raw_models/dpt_large_384.pt` `2f21e586477d90cb9624c7eef5df7891edca49a1c4795ee2cb631fd4daa6ca69`; classifier `results/depth_enhanced_classifier.pth` `83f6c63eeef6ede9ce7e2fed47acf0d594ec1f957684ae357f23a6f0dd491457` |
| 6. Inference profile / config | `artps_full_frozen_mars_clf_on_v1` (`reproduction/iac2026/configs/independent_eval_artps_full_frozen_mars.yaml`) |
| 7. Preprocessing | `mars_enhancement_v1`, Real-ESRGAN off |
| 8. Output | `paper/iac2026/figures/fig_qualitative_artps.png` |
| 9. Generation command | `set ARTPS_DATASET_ROOT=<repo>/mars_images` then `python scripts/iac2026/generate_qualitative_figure.py` |
| 10. Score-blind selection | **YES** |
| 11. Test data used | **NO** (`test_opened=false`, `final_test_authorized=false`) |

Additional (not manuscript metrics): frozen image-score aggregation `max_valid_candidate_after_masks`; Priority Buffer / curiosity / diversity **not** applied; device `cuda`; 4 raw detections, 4 valid candidates after masks.

Machine-readable copy: `fig_qualitative_artps.meta.json`.
