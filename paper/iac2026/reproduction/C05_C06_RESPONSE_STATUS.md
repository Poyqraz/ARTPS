# C05 / C06 response and evidence status

```yaml
status_type: audit_observation
author_attestation_status: not_provided
author_verified: false
recorded_from:
  - repository archaeology
  - manuscript claims
  - scanned local paths
real_run_allowed: false
```

This document records the status of unresolved questionnaire fields based on
repository/document archaeology. It is not an author attestation and must not be
represented as author-confirmed historical experiment metadata.

Questionnaire: [AUTHOR_QUESTIONNAIRE_C05_C06.md](AUTHOR_QUESTIONNAIRE_C05_C06.md)
Author attestation template (empty / pending):
[AUTHOR_ATTESTATION_C05_C06.template.md](AUTHOR_ATTESTATION_C05_C06.template.md)
Audit: [C05_C06_DEFINITION_AUDIT.md](C05_C06_DEFINITION_AUDIT.md)
Gaps: [DATASET_MANIFEST_GAPS.md](DATASET_MANIFEST_GAPS.md)
Defs: [`reproduction/iac2026/C05_C06_DEFINITIONS.yaml`](../../../reproduction/iac2026/C05_C06_DEFINITIONS.yaml)

Independent (non-historical) evaluation protocol pin (does **not** unlock C05/C06):
[INDEPENDENT_EVALUATION_PROTOCOL.md](INDEPENDENT_EVALUATION_PROTOCOL.md) /
[`reproduction/iac2026/INDEPENDENT_EVAL_V1.yaml`](../../../reproduction/iac2026/INDEPENDENT_EVAL_V1.yaml).

Ledger support remains: `accepted_abstract_reproduction_pending`.

**Default evidence_scope for “not located” statements below:**
- git history (including baseline `8f7e3ff` and `git log -S` / `-G` scans)
- scanned repository paths under HEAD
- listed local result / image paths (`results/`, `mars_images/`, fixtures)
- manuscript/PDF sources (`docs/paper.tex`, Full/portal abstracts)

Absence outside the scanned sources is not established.

---

## P0-01 — Evaluation unit

**Answer:** unknown (image-level / pixel-level / region-level not located)

**Source / file path:**
- [C05_C06_DEFINITION_AUDIT.md](C05_C06_DEFINITION_AUDIT.md) field `task_level`
- `8f7e3ff:docs/paper.tex` abstract — “anomaly discrimination (AUROC/AUPRC/F1/FPR)” without sample unit
- Full MS metrics narrative (same ambiguity)

**Confidence:** unknown

---

## P0-02 — Positive and negative class

**Positive class:** unknown

**Negative class:** unknown

**Labeling rule:** unknown

**Source / file path:** no claim-dataset annotation artifact was recovered from the documented scan scope.
Note: `reproduction/iac2026/fixtures/synthetic_manifest.csv` / synthetic config use
`positive_label: 1` / `anomaly_binary` for **software verification only** — not claim GT.

**Confidence:** unknown

---

## P0-03 — Per-file split

**Split file:** not located in the scanned sources

**Path or archive:** no artifact was recovered from the documented scan scope

**evidence_scope:**
- git history
- scanned repository paths
- listed local result paths
- manuscript/PDF sources

**1247 / 892 / 708 numbers mean:** unsure —
manuscript aggregates (Curiosity 1247, Perseverance 892, test/validation bullet 708;
total stated 2847). Whether these are disjoint partition buckets vs mission source counts
overlapping the 708 pool remains **UNKNOWN** (see [DATASET_MANIFEST_GAPS.md](DATASET_MANIFEST_GAPS.md)).

**Confidence:** unknown

Absence outside the scanned sources is not established.

---

## P0-04 — F1 threshold

**Selection split:** validation (manuscript protocol claim only)

**Selection metric:** unknown

**Tie-break:** unknown

**Numeric threshold:** unknown

**Source / log / notebook:**
- `8f7e3ff:docs/paper.tex` Protocol: “Decision thresholds are selected on validation data;
  results are reported on the test set.”
- No logged threshold float / notebook cell recovered from the documented scan scope

**Confidence:** selection-split narrative **likely** (as manuscript claim); metric / tie-break / numeric **unknown**

---

## P0-05 — AUPRC 0.847 method

**Method:** unknown

**Library or function:** unknown

**Source:**
- Accepted abstract / `paper/iac2026/main.tex` — labels the number **AUPRC**
- `docs/Full_Baydemir_ARTPS.pdf` §4.3.1 — “area under the Precision–Recall curve”
  (does not distinguish `average_precision` vs `trapezoidal_pr_auc`)
- Chart hard-codes of 0.847 are **VISUALIZATION_CONSTANT**, not method evidence

**Confidence:** unknown

---

## P0-06 — Baseline 0.856 identity

**Identity:** unknown

**Same test IDs as ARTPS:** unsure
(protocol claims identical train/validation/test splits; no per-file ID intersection proof
was recovered from the documented scan scope)

**Source:**
- Table header “PaDiM/PatchCore (WRN-50-2)” with AUROC 0.856 —
  `8f7e3ff:docs/paper.tex` / Full PDF Table 2 / accepted abstract
- Backbone name WRN-50-2 is a **manuscript claim only**; PaDiM-only vs PatchCore-only vs
  best-of-two vs mean **not located** (HEAD stubs forbid inventing a mean)

**Confidence:** unknown

---

## P0-07 — Predictions or re-inference

**ARTPS predictions:** not located in the scanned sources (no claim-era prediction CSV recovered)

**Baseline predictions:** not located in the scanned sources

**evidence_scope:**
- git history
- scanned repository paths
- listed local result paths
- manuscript/PDF sources

**Checkpoint paths (PATH_REFERENCE_ONLY — presence ≠ provenance):**
- `results/padim_stats.pth` (local; gitignored; SHA/recipe unverified)
- `results/patchcore_bank.pth` (local; gitignored; SHA/recipe unverified)

**Config/notebook/script paths for the published table:** not located in the scanned sources

**Split file:** not located in the scanned sources

**Backup / archive / Drive / old workspace scan (within documented scope):**
- full git history (`git log -S` / `-G`, baseline `8f7e3ff`)
- `stash@{0}` name listing only (not applied)
- local `results/`, `mars_images/`, `docs/*.pdf`, HEAD `reproduction/iac2026/fixtures/`
No C05/C06 claim prediction table or pinned split archive was recovered from that scope.
Synthetic fixture predictions remain SW-only.

**Confidence:** **likely** regarding non-recovery within scanned sources

Absence outside the scanned sources is not established.

---

## Effect on readiness

```text
python scripts/iac2026/check_c05_c06_definition_readiness.py
→ readiness: blocked
→ real_run_allowed: false
→ author_attestation_status: pending
→ author_verified: false
```

P0 fields in `C05_C06_DEFINITIONS.yaml` stay `UNKNOWN_EVIDENCE_NOT_LOCATED` /
`PARTIALLY_LOCATED` as before. This audit-observation document does not close them.
