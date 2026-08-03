# Depth semantics (IAC / paper glossary)

Shipped ARTPS does **not** claim metric distance. Use this wording in Results/Methods:

1. **Relative depth ordering** — Monocular DPT (or fallback CNN) produces a within-image map. `0=Near`, `1=Far` only ranks pixels in that image; not calibrated metres; not comparable across unrelated images.
2. **Apparent size / relative size–distance features** — Proposal gates use `apparent_size` and related features from image area + relative far. This is not metric (physical) size.
3. **Image-relative near/far categorization** — `size_distance_band` (`near_small`, `far_small`, …) is a within-image policy band, not an absolute near/far class label.
4. **Calibration hook** — `estimate_depth_scale_m` returns `None` until scale-aware depth exists; `metric_size` stays unset without a scale.

See also: [README Depth output semantics](../../README.md#depth-output-semantics).
