# Close vs distant qualitative Fig. 4 provenance

Illustration only. Not a quantitative experiment, not a near/far benchmark, and not a Results metric.

| # | Field | Value |
| --- | --- | --- |
| 1 | All candidate paths considered | Close 1 `train/hills_or_ridge/curiosity_1100_MAST_938_jpg.rf.7417a3036ec4af81b3b9d4305c05eee3.jpg`; Close 2 `train/boulder/percy_sol1450_MCZ_RIGHT_9_jpg.rf.f390f8c84becbe615a34db73d9f2610e.jpg`; Far 1 `train/flat_terrain/curiosity_1100_MAST_827_jpg.rf.fd10bd35d413cba7432b79ab8433e9b6.jpg`; Far 2 `train/rocky/curiosity_1100_MAST_817_jpg.rf.7d755ad9d3fcbac273a3dfffdc0b3c40.jpg`; Far 3 `train/rover/percy_sol150_NAVCAM_LEFT_8_jpg.rf.5d964d0db273d6db4a7054ec8516c688.jpg` |
| 2 | Selected close image ID / filename | `author_pool_close1_curiosity_1100_MAST_938_jpg.rf.7417a3036ec4af81b3b9d4305c05eee3.jpg` / `curiosity_1100_MAST_938_jpg.rf.7417a3036ec4af81b3b9d4305c05eee3.jpg` |
| 3 | Selected far image ID / filename | `ie1_train_rocky_curiosity_1100_MAST_817_jpg.rf.7d755ad9d3fcbac273a3dfffdc0b3c40.jpg_1732bef60212` / `curiosity_1100_MAST_817_jpg.rf.7d755ad9d3fcbac273a3dfffdc0b3c40.jpg` |
| 4 | Split for each selected image | close: filesystem **train** (not in `independent_eval_v1` included rows); far: manifest **train**, included |
| 5 | Repository commit SHA | `947743fb7520bfca14c81605ac7dce4b97b47edb` (`main` @ PR #40) |
| 6 | Inference profile | `artps_full_frozen_mars_clf_on_v1` (`reproduction/iac2026/configs/independent_eval_artps_full_frozen_mars.yaml`) |
| 7 | Checkpoint identities / hashes | AE `results/optimized_autoencoder_curiosity_extended.pth` `8186bbe6be424dd212d5d4a93b1ae36b80939552519706b4a8680c5d05e995f2`; DPT_Large `raw_models/dpt_large_384.pt` `2f21e586477d90cb9624c7eef5df7891edca49a1c4795ee2cb631fd4daa6ca69`; classifier `results/depth_enhanced_classifier.pth` `83f6c63eeef6ede9ce7e2fed47acf0d594ec1f957684ae357f23a6f0dd491457` |
| 8 | Preprocessing profile | `mars_enhancement_v1`, Real-ESRGAN off |
| 9 | Generation command | `set ARTPS_DATASET_ROOT=<repo>/mars_images` then `python scripts/iac2026/generate_close_far_qualitative_figure.py` |
| 10 | Output file path | `paper/iac2026/figures/fig_close_far_qualitative_artps.png` |
| 11 | Test used | **NO** (`test_opened=false`, `final_test_authorized=false`) |
| 12 | Author-provided candidate pool only | **YES** |
| 13 | Agent-selected outside pool | **NO** |
| 14 | Quantitative experiment | **NO** |
| 15 | Qualitative selection rationale | Close 1 + Far 2: Curiosity Mastcam pair with readable post-suppression combined maps and valid-candidate overlays. Close 2 rejected: Perseverance MCZ domain mismatch vs manuscript Curiosity-Mastcam framing; overlay misses the two most salient near rocks. Far 1 rejected: valid overlay includes a full-width top-edge frame artefact. Far 3 rejected: rover hardware still dominates RGB; map is sparser than Far 2; not cleaner than Far 1/2. Selection was visual, not score-maximization. |

Additional: panel (b)/(e) = returned `combined_map` (post-suppression/refinement), same semantics as Fig.~2(b); overlay = `_score_object_detections` kept boxes only; Priority Buffer / curiosity / diversity **not** applied; close file SHA256 `4b09d26fdc90e154c661720397277d215e0646a4238893bf46aa5dc278d5d88c`; far file SHA256 `1732bef60212e10724fb6005d5edfe87fff9000b6f08297d7ce971de2a549f8d`.

Machine-readable copy: `fig_close_far_qualitative_artps.meta.json`.
