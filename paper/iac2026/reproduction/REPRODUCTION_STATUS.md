# Reproduction status (C05–C07 harness)

Statuses must not become `measured` until real dataset + exact config + real run + raw outputs are pinned. Synthetic tests = software verification only.

| Claim | Harness | Historical definition recovered | Dataset pinned | Config pinned | Real run completed | Raw output pinned | Status |
|------|---------|---------------------------------|----------------|---------------|--------------------|-------------------|--------|
| C05 | prediction-table metrics runner | Partial (counts + val-threshold note); task/score/split **UNKNOWN** | No | Example only (`TASK_LEVEL_TBD`) | No | No | `accepted_abstract_reproduction_pending` |
| C06 | fail-loud PaDiM/PatchCore adapters | Baseline identity **UNKNOWN** | No | Example only | No | No | `accepted_abstract_reproduction_pending` |
| C07 | `scripts/benchmark_cv_core_speed.py` | Script recovered historically; 28.1 pin **UNKNOWN** | N/A | Example YAML | No (harness only) | No | `accepted_abstract_reproduction_pending` |
| C08 | protocol only | Planned Jetson path | No | No | No | No | `planned` |

See [ARCHAEOLOGY_REPORT.md](ARCHAEOLOGY_REPORT.md) for blockers.
