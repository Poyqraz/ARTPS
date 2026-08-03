# Depth perception — status and future work

## Shipped (current claims)

ARTPS ships **monocular relative depth ordering** only (within-image near/far ranks). See [results/paper_figs/depth_semantics.md](results/paper_figs/depth_semantics.md).

- Local DPT_Large weights when present (`raw_models/dpt_large_384.pt`); otherwise Hub or lightweight fallback CNN
- Depth cues feed enhancement, anomaly fusion, visualization QC, and **apparent-size** / image-relative size–distance policy bands
- `estimate_depth_scale_m(...)` returns `None`; `metric_size` stays unset without a scale

## Not claimed

Do not state or imply in UI, docs, or paper:

- Metric / calibrated distance (metres)
- Cross-image absolute depth comparison
- Stereo disparity / point clouds as shipped capability
- Metric 3D size, volume, or physical object dimensions

## Future work (out of current manuscript claims)

- Scale-aware depth calibration hook (populate `estimate_depth_scale_m`)
- Stereo / multi-view depth if mission hardware provides it
- Metric size only after a validated scale exists

Category auto-labeling and hybrid-model extensions remain separate product backlog items; they are not IAC Results claims until measured under a documented protocol.
