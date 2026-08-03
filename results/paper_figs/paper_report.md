# ARTPS paper figure summary

- Sample count: 5
- Curiosity mean / std: 0.707 / 0.108
- AE MSE mean / std: 0.008343 / 0.004375
- Depth variance mean / std: 0.083996 / 0.005591
- Roughness mean / std: 0.036435 / 0.020033

## Distributions and relationships

![Dataset Summary](dataset_summary.png)

![Correlation heatmap](corr_heatmap.png)

## Highest-curiosity examples (Top-5)

![Top Grid](topk_grid.png)

## Lowest-curiosity examples (Bottom-5)

![Bottom Grid](bottomk_grid.png)

## Detection overlay examples

Boxed overlays from the fused anomaly map (AE residual + depth edge + texture/shadow + optional PaDiM/PatchCore cues). Target prioritization framing aligns with rover autonomy literature [Estlin et al., 2014] ([JPL ISAIRAS](https://ai.jpl.nasa.gov/public/documents/papers/estlin-isairas2014-automated.pdf)).

![0735MR0031500040403079E01_DXXX_det_overlay.png](detection_overlays/0735MR0031500040403079E01_DXXX_det_overlay.png)

![FRF_0940_0750382098_770ECM_N0460000FHAZ00206_01_295J_calib01_areo.info_det_overlay.png](detection_overlays/FRF_0940_0750382098_770ECM_N0460000FHAZ00206_01_295J_calib01_areo.info_det_overlay.png)

![curiosity_0000_Sol_958__Mast_Camera_(Mastcam)_det_overlay.png](detection_overlays/curiosity_0000_Sol_958__Mast_Camera_(Mastcam)_det_overlay.png)

![curiosity_1100_MAST_1460_jpg.rf.f546b807109c1df632cb62e069ded089_det_overlay.png](detection_overlays/curiosity_1100_MAST_1460_jpg.rf.f546b807109c1df632cb62e069ded089_det_overlay.png)

![curiosity_1100_NAVCAM_540_jpg.rf.6315c37bd960ce862e4c6161408009cf_det_overlay.png](detection_overlays/curiosity_1100_NAVCAM_540_jpg.rf.6315c37bd960ce862e4c6161408009cf_det_overlay.png)

## Detector benchmark matrix

**Depth semantics:** ARTPS uses monocular **relative depth ordering** and **apparent size** / image-relative near–far bands — not metric distance or metric size (see [depth_semantics.md](depth_semantics.md)).

Read detector comparisons with task-oriented metrics, not only aggregate accuracy:

- `proposal_recall`: fraction of science targets hit by at least one box
- `bbox_precision`: fraction of boxes that are on-target
- `small_object_recall`: recall on small / far-looking targets (image-relative)
- `false_positive_rate`: rover-body, horizon-band, and border-shadow false alarms
- `avg_detections_per_image`: operational load / box density
- `ranking_quality`: object-level anomaly/value ranking quality

Suggested comparison axes:

1. `heuristic`: contour + anomaly fusion baseline
2. `yolo`: optional ONNX detector, rescored with object-level scorer
3. `rt_detr`: second-phase benchmark candidate
4. `segmentation_first`: mask-to-box + anomaly segmentation research track

## Literature-driven integration notes

- Industrial visual anomaly surveys: separating heatmap quality from object proposal layers eases field tuning.
- Open-vocabulary detectors may help future class flexibility; classic YOLO-like boxes remain lower integration risk for a first step.
- Segmentation-first / reasoning anomaly lines can help rare small targets in discovery settings.

Recommended production sequence:

1. Freeze heuristic baseline via export + benchmark
2. Generate weak-supervision pseudo-labels
3. Attach a YOLO-like detector as proposal backend
4. Rank with per-box anomaly/value fusion
