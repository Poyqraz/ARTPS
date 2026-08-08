# Evidence map: historical vs repo vs independent_eval_v1_1

status: research_complete
purpose: manuscript integration only (no new experiment)
pr29_merge: d939fd373b04e26371a12ce0fab8c9f80ce3b7a2
branch: paper/iac2026-v1-1-integration

These three categories must not be conflated in the IAC 2026 manuscript.

## A. Historical manuscript / accepted abstract

| Metric | Value | Ledger | Meaning |
|---|---|---|---|
| AUROC | 0.894 | C05 `accepted_abstract_reproduction_pending` | Abstract claim; repo reproduction pending |
| AUPRC | 0.847 | C05 same | not “historical result is false” |
| F1 | 0.823 | C05 same | |
| Baseline AUROC | 0.856 | C06 same | identity unrecovered |
| FPS @ 256×256 core | 28.1 | C07 same | lightweight / no learned depth+AE |

Sources: accepted IAC abstract; [`paper/iac2026/main.tex`](../main.tex) abstract (unchanged in this PR); Table 1 `tab:accepted-abstract` in [`sections/results.tex`](../sections/results.tex).

## B. Current repository-supported implementation

- Frozen ARTPS profile `artps_full_frozen_mars_clf_on_v1`, FP32.
- Pinned validation predictions already on main; this PR does not retrain, retune, flip orientation, or re-run inference.
- `independent_eval_v1` manifest immutable (LF SHA `9f953dc0…`).
- `test_opened: false`, `final_test_authorized: false`.
- Claim `IND_EVAL_V1` stays `protocol_defined_pending_data` (test still closed).

## C. Supplementary independent_eval_v1_1 evidence

Human-reviewed 360-sample annotation layer + frozen-score validation relabel.

| Item | Value |
|---|---|
| Overall labels | 340 positive / 20 negative |
| Train / val / test | 242/10 · 46/8 · 52/2 |
| Label audit | 360 reviewed; 188 agree / 172 disagree vs v1 heuristic |
| Validation AUROC (principal) | 0.772 |
| Validation AP (secondary; prevalence 46/54≈0.852) | 0.956 |
| Validation F1 (diagnostic only) | 0.920 at threshold 0.0 (TN=0 FP=8 FN=0 TP=46; all-positive) |
| Test | unopened / unreported |

Pinned artifacts: `reproduction/iac2026/manifests/independent_eval_v1_1.csv` + `.meta.json`; `annotations/independent_eval_v1_1_review_provenance.csv` + `.meta.json`; `INDEPENDENT_EVAL_V1_1_REPORT.md`.

This is a **current reproducible supplementary validation**, not a reproduction of 0.894, not a replacement or correction of the historical experiment, and not evidence that the original paper failed.

## Integration rule for this PR

- Abstract: do not edit; do not replace 0.894 with 0.772.
- Results: keep Table 1 historical; add a **separate** supplementary table (`tab:indep-v11`).
- Headline supplementary metric: AUROC 0.772. F1 0.920 is not a headline discriminative claim.
- AI declaration unchanged; no funding/grant text.
