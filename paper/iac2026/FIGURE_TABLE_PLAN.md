# Figure and table inventory + experiment map

Do **not** invent final Result numbers in LaTeX until ledger support is closed.

## Planned tables

| ID | Title | Experiment | Ledger | Status |
|----|-------|------------|--------|--------|
| T_detect | Detection AUROC / AUPRC / F1 vs baseline | Primary detection bench | C05–C06 | planned |
| T_hw | Workstation FPS / latency by resolution | Speed bench | C07–C08 | planned |
| T_shadow_proxy | Shadow / FP proxy OFF vs ON | Shadow proxy | C09–C10 | artifact on disk |
| T_sd_proxy | Size–distance policy proxy | SD proxy | C11 | artifact on disk |
| T_jetson | Jetson FPS / memory by profile | Jetson protocol | C14 | planned |
| T_ops | Priority Buffer / diversity (if measured) | Operational study | C03–C04 | planned |

## Planned figures

| ID | Content | Notes |
|----|---------|-------|
| F_pipeline | ARTPS block diagram | Methods |
| F_depth_rel | Relative depth map example | Caption: not metric |
| F_overlays | Detection overlays | From `results/paper_figs/detection_overlays/` if used |
| F_proxy_shadow | Optional bar chart for T_shadow_proxy | From JSON |
| F_proxy_sd | Optional bar chart for T_sd_proxy | From JSON |

## On-disk proxy artifacts (inventory only this PR)

| Artifact | Maps to |
|----------|---------|
| `results/paper_figs/iac_shadow_proxy_summary.json` | T_shadow_proxy |
| `results/paper_figs/iac_shadow_proxy_table.md` | T_shadow_proxy |
| `results/paper_figs/iac_size_distance_proxy_summary.json` | T_sd_proxy |
| `results/paper_figs/iac_size_distance_proxy_table.md` | T_sd_proxy |
| `results/paper_figs/depth_semantics.md` | caption language |
| `results/paper_figs/paper_report.md` | exploratory figs (not IAC final) |

## Experiment → table map

```text
Primary detection bench -----------> T_detect (C05–C06)
Workstation speed -----------------> T_hw (C07)
run_iac_shadow_proxy_eval.py ------> T_shadow_proxy (C09–C10)
run_iac_size_distance_proxy_eval.py > T_sd_proxy (C11)
test_size_distance_lite_bench.py --> (no paper table; software_verification)
Jetson protocol -------------------> T_jetson (C14)
```
