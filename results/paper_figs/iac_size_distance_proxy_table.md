# IAC size/distance policy proxy ablation

Proxy metrics (class labels, OFF-run far-small pseudo-GT, self-IoU). Not human bbox GT. Lite bench is software verification only - not a performance result. Bands are **image-relative near/far categorization**; features use **apparent size**, not metric size or calibrated distance. See [depth_semantics.md](depth_semantics.md).

| Metric | Policy OFF | Policy ON | Delta |
|--------|------------|-----------|-------|
| far-small recall (proxy) | — | None | (ON vs OFF pseudo-GT) |
| near-large over-merge | 0.0 | 0.0 | 0.0 |
| field-scale FPR | 0.4999999999999999 | 0.35119047619047616 | -0.14880952380952372 |
| mean matched IoU (self) | — | 0.9985775248933143 | (not GT localization) |
| avg detections | 2.2857142857142856 | 2.761904761904762 | — |

n_images=21

Note: `tests/test_size_distance_lite_bench.py` is software verification only — not a performance result.
