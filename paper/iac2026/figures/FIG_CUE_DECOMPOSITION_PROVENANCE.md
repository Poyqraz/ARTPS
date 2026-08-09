# Cue-decomposition Fig. 3 provenance

Illustration only. Not a quantitative experiment and not a Results metric.
Source selection is inherited from Fig. 2; no new sample was chosen.

| # | Field | Value |
| --- | --- | --- |
| 1 | Source image ID | `ie1_train_boulder_curiosity_100_MAST_106_jpg.rf.78c4b8fcf62f2b932ebccf43bd427980.jpg_4dd2d929c0ee` |
| 2 | Manifest product_id | `curiosity_100_mast_106` |
| 3 | Relative path | `train/boulder/curiosity_100_MAST_106_jpg.rf.78c4b8fcf62f2b932ebccf43bd427980.jpg` |
| 4 | File SHA256 | `4dd2d929c0ee66f91b17c3f73b8bf127f23b6f6775511da3a735e927fbf2291a` |
| 5 | Split | `train` (not test) |
| 6 | Source selection inherited from Fig. 2 | **YES** (`fig_qualitative_artps.meta.json`; generator does not call `select_sample()`) |
| 7 | New sample selection | **NO** |
| 8 | Repository commit at generation | `e1607508375ac841b68252e548c7adf7440e81bd` (`main` @ PR #39); panel (d) relabel only |
| 9 | Checkpoints | AE `results/optimized_autoencoder_curiosity_extended.pth` `8186bbe6be424dd212d5d4a93b1ae36b80939552519706b4a8680c5d05e995f2`; DPT_Large `raw_models/dpt_large_384.pt` `2f21e586477d90cb9624c7eef5df7891edca49a1c4795ee2cb631fd4daa6ca69`; classifier `results/depth_enhanced_classifier.pth` `83f6c63eeef6ede9ce7e2fed47acf0d594ec1f957684ae357f23a6f0dd491457` |
| 10 | Inference profile / config | `artps_full_frozen_mars_clf_on_v1` (`reproduction/iac2026/configs/independent_eval_artps_full_frozen_mars.yaml`) |
| 11 | Preprocessing | `mars_enhancement_v1`, Real-ESRGAN off |
| 12 | Frozen fusion weights | primary `w_recon=0.50`, `w_depth=0.30`, `w_texture=0.20`; also in the mix (not extra panels) `w_lap=0.08` (`depth_lap_n`), `w_detail=0.12` (gray Laplacian+DoG) |
| 13 | Inference vs visualization normalization | inference: `_normalize_map` percentile 2–98 per cue before weighting; visualization: per-panel min–max for display only |
| 14 | Output | `paper/iac2026/figures/fig_cue_decomposition_artps.png` |
| 15 | Generation command | `set ARTPS_DATASET_ROOT=<repo>/mars_images` then `python scripts/iac2026/generate_cue_decomposition_figure.py` |
| 16 | Test data used | **NO** (`test_opened=false`, `final_test_authorized=false`) |
| 17 | Quantitative experiment | **NO** |
| 18 | Classifier in fused map | **NO** (classifier ON in config for scoring / curiosity; not a `combined_map` term) |

Additional: score-based cherry-picking **NO**; Priority Buffer / curiosity / diversity **not** applied; panel (b) is relative-depth **edge** (Sobel magnitude), not protrusion; panel (d) is `raw_combined_pre_mask` after proximity mix and **before** FP suppression (display title: pre-suppression fused map); cue arrays come from `compute_combined_anomaly_map` diagnostics (`recon_diff_n`, `depth_edge_n`, `texture_term`, `raw_combined_pre_mask`). Display min–max is per panel and not numerically comparable across panels.

Machine-readable copy: `fig_cue_decomposition_artps.meta.json`.
