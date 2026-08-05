# Accuracy sprint plan (independent_eval_v1 → manuscript Results)

Timeline focused on **data → model runs → Results**. This document must not contain
funding, grant, sponsorship, or budget application language.

## Calendar

| Dates | Work |
|-------|------|
| 6–7 Aug | Dataset inventory; primary domain selection; annotation queue |
| 8–10 Aug | Primary annotation (workspace visual review); uncertain/exclusion handling |
| 11 Aug | Annotation quality re-review; duplicate/scene grouping |
| 12 Aug | Real SHA-pinned manifest; split-ratio lock; deterministic split freeze |
| 13 Aug | Dataset SHA/leakage audit; `ready_for_model_runs` decision |
| 14–16 Aug | **Next PR:** ARTPS full/lightweight + PaDiM + PatchCore on validation |
| 17 Aug | Frozen test run (single open) |
| 18–20 Aug | Results/Discussion integration into full paper (evidence-gated) |

## Hard freezes

- Abstract / C05–C07 numbers remain `accepted_abstract_reproduction_pending` until separately closed.
- `IND_EVAL_V1` ledger support stays `protocol_defined_pending_data` until metrics artifacts exist.
- Do not tune thresholds on the frozen test split.
- Do not invent Results cells before runs.

## Pointers

- Domain: [`reproduction/INDEPENDENT_EVAL_V1_DOMAIN_SELECTION.md`](reproduction/INDEPENDENT_EVAL_V1_DOMAIN_SELECTION.md)
- Manifest: `reproduction/iac2026/manifests/independent_eval_v1.csv`
- Readiness: `results/iac2026/dataset_build/dataset_readiness.json`
