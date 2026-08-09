# Qualitative Fig. 2 provenance

Illustration only. Not a quantitative experiment and not a Results metric.

| Field | Value |
| --- | --- |
| 1. Source image ID | `author_pool_AUTHOR_1_curiosity_300_MAST_453_jpg.rf.6ecd29659d982741653bbe91b11ef22b.jpg` |
| Manifest product_id | `curiosity_300_mast_453` (not in `independent_eval_v1` included rows) |
| Relative path | `train/boulder/curiosity_300_MAST_453_jpg.rf.6ecd29659d982741653bbe91b11ef22b.jpg` |
| Mission / instrument | Curiosity / Mastcam |
| File SHA256 | `a94b785f4e9cf88fcf07ae7ddabe5b79c53957afe764aac3b14ba3125d7571a2` |
| 2. Split | filesystem `train` (not test) |
| 3. Selection procedure | selected before inference from an author-provided RGB candidate pool using qualitative scene-composition and domain criteria (`AUTHOR_1`); pool also considered AUTHOR_2–4; no optional dataset search |
| 4. Selection lock | `source_selection_locked=true`; origin `AUTHOR_1` |
| 5. Repository commit at generation | `acb328619376618b530d97bd041e64d59209b538` (`main` @ PR #41) |
| 6. Checkpoints | AE `results/optimized_autoencoder_curiosity_extended.pth` `8186bbe6be424dd212d5d4a93b1ae36b80939552519706b4a8680c5d05e995f2`; DPT_Large `raw_models/dpt_large_384.pt` `2f21e586477d90cb9624c7eef5df7891edca49a1c4795ee2cb631fd4daa6ca69`; classifier `results/depth_enhanced_classifier.pth` `83f6c63eeef6ede9ce7e2fed47acf0d594ec1f957684ae357f23a6f0dd491457` |
| 7. Inference profile / config | `artps_full_frozen_mars_clf_on_v1` (`reproduction/iac2026/configs/independent_eval_artps_full_frozen_mars.yaml`) |
| 8. Preprocessing | `mars_enhancement_v1`, Real-ESRGAN off |
| 9. Output | `paper/iac2026/figures/fig_qualitative_artps.png` |
| 10. Generation command | `set ARTPS_DATASET_ROOT=<repo>/mars_images` then `python scripts/iac2026/generate_qualitative_figure.py` |
| 11. `model_output_used_for_selection` | **false** |
| 12. `score_based_cherry_picking` | **false** |
| 13. Test data used | **NO** (`test_opened=false`, `final_test_authorized=false`) |
| Overlay visualization | `candidate_support_v1` (open-corner ROI + proposal-support contour + combined-map anchor) |
| `support_geometry_source` | proposal hysteresis/CC contour persisted as visualization-only metadata; no new map threshold |
| `anchor_definition` | argmax of post-suppression `combined_map` inside support contour, else ROI; `peak_xy` if present |
| `fallback_behavior` | open-corner ROI + anchor when no proposal CC survives |
| `visualization_only` | **true** |
| `candidate_scores_changed` | **false** |
| `validity_decisions_changed` | **false** |
| `image_scores_changed` | **false** |
| `quantitative_experiment` | **false** |

Additional (not manuscript metrics): frozen image-score aggregation `max_valid_candidate_after_masks`; Priority Buffer / curiosity / diversity **not** applied; device `cuda`; 3 raw detections, 2 valid candidates after masks; overlay geometry `n_support_contour=2`, `n_oriented_poly=0`, `n_bracket_fallback=0`. Overlay panel is a candidate-support overlay, not a segmentation.

Machine-readable copy: `fig_qualitative_artps.meta.json`.
