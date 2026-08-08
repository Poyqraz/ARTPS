# Figure and table inventory + experiment map

Do **not** invent final Result numbers in LaTeX until ledger support is closed. Accepted-abstract numbers (C05–C07) stay `accepted_abstract_reproduction_pending`. Independent eval (`IND_EVAL_V1`) stays `protocol_defined_pending_data` until the test split is opened. Supplementary `IND_EVAL_V1_1` validation audit is a separate table (`tab:indep-v11`), not a substitute for Table 2. Pending tables stay **plan-only** (no empty numeric cells in TeX).

## Tables (full-paper v0.1 map)

| ID | Title | TeX now? | Ledger | Status |
|----|-------|----------|--------|--------|
| Table 1 | Reported results from the accepted IAC abstract | Yes (`tab:accepted-abstract`) | C05–C07 | abstract reference + reproduction-pending footnote |
| Table 2 | Independent eval detection (AUROC/AUPRC/F1) | No — plan only | IND_EVAL_V1 | protocol_defined_pending_data (test closed) |
| Table 2s | Supplementary v1→v1_1 label audit + frozen-score validation | Yes (`tab:indep-v11`) | IND_EVAL_V1_1 | measured validation-only; AUROC 0.772 principal; F1 0.920 prevalence-dependent/all-positive; no test |
| Table 3 | Independent eval vs PaDiM/PatchCore baselines | No — plan only | IND_EVAL_V1 | pending metrics |
| Table 4 | Workstation FPS / latency by resolution | No — plan only | C07 / C08 | reproduction-pending / planned |
| Table 5 | Shadow / FP proxy OFF vs ON | No — plan only | C09–C10 | preliminary proxy (non-headline) |
| Table 6 | Size–distance policy proxy | No — plan only | C11 | preliminary proxy (non-headline) |

Legacy IDs (`T_detect`, `T_hw`, …) map to Tables 2–6 above.

## Figures (full-paper v0.1 map)

| ID | Content | TeX now? | Notes |
|----|---------|----------|-------|
| Fig 1 | ARTPS pipeline schematic | Yes (`fig:pipeline`) | `figure*` tabular placeholder; no missing `\includegraphics` |
| Fig 2 | Relative depth map example | No — plan only | Caption must say not metric distance |
| Fig 3 | Detection overlays | No — plan only | From artifacts when available |
| Fig 4 | Optional shadow/FP proxy chart | No — plan only | Preliminary proxy only |
| Fig 5 | Optional size–distance proxy chart | No — plan only | Preliminary proxy only |

## On-disk artifacts (inventory)

| Artifact | Maps to | Role |
|----------|---------|------|
| `results/paper_figs/iac_shadow_proxy_summary.json` | Table 5 / Fig 4 | preliminary proxy |
| `results/paper_figs/iac_shadow_proxy_table.md` | Table 5 | preliminary proxy |
| `results/paper_figs/iac_size_distance_proxy_summary.json` | Table 6 / Fig 5 | preliminary proxy |
| `results/paper_figs/iac_size_distance_proxy_table.md` | Table 6 | preliminary proxy |
| `results/paper_figs/depth_semantics.md` | Fig 2 caption language | claim boundary |
| `results/paper_figs/paper_report.md` | — | **exploratory / qualitative only**; **not** a quantitative manuscript result |

## Experiment → table map

```text
Accepted abstract (historical) -----> Table 1 (reference only; reproduction pending)
independent_eval_v1 ---------------> Tables 2–3 (pending data; test closed)
independent_eval_v1_1 -------------> Table 2s / tab:indep-v11 (supplementary validation only)
Workstation speed -----------------> Table 4 (reproduction pending / planned)
run_iac_shadow_proxy_eval.py ------> Table 5 (proxy, non-headline)
run_iac_size_distance_proxy_eval.py > Table 6 (proxy, non-headline)
Jetson protocol -------------------> (future; C08/C14 planned; not Table 1–6 numeric yet)
paper_report.md -------------------> not an IAC Results source
```
