# Reproduction status (C05–C07 harness)

Statuses must not become `measured` until real dataset + exact config + real run + raw outputs are pinned and registered. Synthetic tests = software verification only.

| Claim | Harness infrastructure | Historical definition recovered | Dataset pinned | Config pinned | Real run completed | Raw output pinned | Ledger support (unchanged) |
|------|------------------------|---------------------------------|----------------|---------------|--------------------|-------------------|----------------------------|
| C05 | prediction-table metrics + fail-closed audit (`implemented` / software-verification infrastructure) | Partial; deep audit in `C05_C06_DEFINITION_AUDIT.md` — task/score/split/pr_method still **UNKNOWN** | No (template only: `manifests/c05_c06_manifest.template.csv`) | Example + synthetic + `C05_C06_DEFINITIONS.yaml` | No | No | `accepted_abstract_reproduction_pending` |
| C06 | fail-loud PaDiM/PatchCore stubs | Baseline identity **UNKNOWN** (questionnaire P0) | No (same template) | Example only + definitions YAML | No | No | `accepted_abstract_reproduction_pending` |
| C07 | historical_exact + current_enhancement_historical_surrogate (`implemented`; equivalence = regression_smoke / not_independently_verified) | Script recovered @ 8f7e3ff surrogate; 28.1 pin **UNKNOWN**; real inputs require manifest+SHA | N/A | Example YAMLs (manifest-pinned) | No (harness only) | No | `accepted_abstract_reproduction_pending` |
| C08 | protocol only | Planned Jetson path | No | No | No | No | `planned` |

See [ARCHAEOLOGY_REPORT.md](ARCHAEOLOGY_REPORT.md) for blockers.
