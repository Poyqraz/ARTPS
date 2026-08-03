# Reproduction status (C05–C07 harness)

Statuses must not become `measured` until real dataset + exact config + real run + raw outputs are pinned and registered. Synthetic tests = software verification only.

| Claim | Harness infrastructure | Historical definition recovered | Dataset pinned | Config pinned | Real run completed | Raw output pinned | Ledger support (unchanged) |
|------|------------------------|---------------------------------|----------------|---------------|--------------------|-------------------|----------------------------|
| C05 | prediction-table metrics + fail-closed audit (`implemented` / software-verification infrastructure) | Partial; task/score/split/pr_method **UNKNOWN** | No | Example + synthetic | No | No | `accepted_abstract_reproduction_pending` |
| C06 | fail-loud PaDiM/PatchCore stubs | Baseline identity **UNKNOWN** | No | Example only | No | No | `accepted_abstract_reproduction_pending` |
| C07 | historical_exact + current_production profiles (`implemented` but equivalence pending for 28.1) | Script recovered @ 8f7e3ff surrogate; 28.1 pin **UNKNOWN** | N/A | Example YAMLs | No (harness only) | No | `accepted_abstract_reproduction_pending` |
| C08 | protocol only | Planned Jetson path | No | No | No | No | `planned` |

See [ARCHAEOLOGY_REPORT.md](ARCHAEOLOGY_REPORT.md) for blockers.
