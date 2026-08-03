# IAC shadow / FP proxy ablation

Proxy metrics (class labels, depth/shadow masks, OFF-run pseudo-GT). Not human bbox GT. Synthetic size/distance tests are software verification only.

| Metric | OFF | ON | Delta |
|--------|-----|----|-------|
| shadow-dense FPR | 0.08333333333333333 | 0.0 | -0.08333333333333333 |
| rover-body FP count | 0 | 1 | — |
| target recall proxy | — | 0.3125850340136055 | (ON vs OFF pseudo-GT) |
| shadow-rock loss | — | 0.0 | (lost OFF shadow-rocks) |
| avg detections | 3.4761904761904763 | 2.761904761904762 | — |

n_images=21, n_shadow_dense=6
