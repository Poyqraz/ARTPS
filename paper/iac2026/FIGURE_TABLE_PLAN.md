# Figure and table inventory + experiment map

Do **not** invent final Result numbers in LaTeX until ledger support is closed. Accepted-abstract numbers (C05–C07) stay `accepted_abstract_reproduction_pending` — no final Results tables in this PR.

## Planned tables

| ID | Title | Experiment | Ledger | Status |
|----|-------|------------|--------|--------|
| T_detect | Detection AUROC / AUPRC / F1 vs baseline | Primary detection bench | C05–C06 | accepted_abstract_reproduction_pending |
| T_hw | Workstation FPS / latency by resolution | Speed bench | C07–C08 | accepted_abstract_reproduction_pending / planned |
| T_shadow_proxy | Shadow / FP proxy OFF vs ON | Preliminary proxy | C09–C10 | artifact on disk (`proxy`) |
| T_sd_proxy | Size–distance policy proxy | Preliminary proxy | C11 | artifact on disk (`proxy`) |
| T_jetson | Jetson FPS / memory / power by profile | Jetson protocol | C14 | planned |
| T_ops | Priority Buffer / diversity (if measured) | Operational study | C03–C04 | planned |

## Planned figures

| ID | Content | Notes |
|----|---------|-------|
| F_pipeline | ARTPS block diagram | Methods |
| F_depth_rel | Relative depth map example | Caption: not metric distance |
| F_overlays | Detection overlays | From `results/paper_figs/detection_overlays/` if used |
| F_proxy_shadow | Optional bar chart for T_shadow_proxy | Preliminary proxy only |
| F_proxy_sd | Optional bar chart for T_sd_proxy | Preliminary proxy only |

## On-disk artifacts (inventory)

| Artifact | Maps to | Role |
|----------|---------|------|
| `results/paper_figs/iac_shadow_proxy_summary.json` | T_shadow_proxy | preliminary proxy |
| `results/paper_figs/iac_shadow_proxy_table.md` | T_shadow_proxy | preliminary proxy |
| `results/paper_figs/iac_size_distance_proxy_summary.json` | T_sd_proxy | preliminary proxy |
| `results/paper_figs/iac_size_distance_proxy_table.md` | T_sd_proxy | preliminary proxy |
| `results/paper_figs/depth_semantics.md` | caption language | claim boundary |
| `results/paper_figs/paper_report.md` | — | **exploratory / qualitative only**; sample count = 5; **not** a quantitative manuscript result |

## Experiment → table map

```text
Primary detection bench -----------> T_detect (C05–C06, reproduction pending)
Workstation speed -----------------> T_hw (C07, reproduction pending)
run_iac_shadow_proxy_eval.py ------> T_shadow_proxy (C09–C10, proxy)
run_iac_size_distance_proxy_eval.py > T_sd_proxy (C11, proxy)
test_size_distance_lite_bench.py --> (no paper table; software_verification)
Jetson protocol -------------------> T_jetson (C14, planned)
paper_report.md -------------------> not an IAC Results source
```
