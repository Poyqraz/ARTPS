# Primary evaluation definition audit

Evidence-only. Do not invent per-file splits, numeric thresholds, or fusion-mode attachment to AUROC 0.894.

Sources: Full PDF / `8f7e3ff:docs/paper.tex`; [`DATASET_MANIFEST_GAPS.md`](reproduction/DATASET_MANIFEST_GAPS.md); [`C05_C06_DEFINITION_AUDIT.md`](reproduction/C05_C06_DEFINITION_AUDIT.md); [`CLAIM_EVIDENCE_LEDGER.md`](CLAIM_EVIDENCE_LEDGER.md) C02/C05/C06; current IAC TeX; `src/models/anomaly/padim.py` / `patchcore.py`.

| Item | Evidence | Source | Confidence | Manuscript-safe? | Notes |
| --- | --- | --- | --- | --- | --- |
| Total images | 2,847 | Full PDF §4.1; 32p Dataset | LEVEL B | YES | Aggregate claim; no SHA-pinned manifest |
| Curiosity Mastcam | 1,247 (Sol 100–1700) | Same | LEVEL B | YES | Mission/instrument count |
| Perseverance Mastcam-Z | 892 (Sol 1–400) | Same | LEVEL B | YES | Mission/instrument count |
| 708 evaluation subset | “test/validation” under diverse field conditions | Same | LEVEL B as subset only | YES as subset; NO as disjoint partition | 1247+892+708=2847 is arithmetic, not proof of three disjoint buckets |
| Resolution span | ~640×480 to 1920×1080 | Dataset section | LEVEL B | YES | Do not add extra numeric results |
| Train/val/test protocol | Identical splits; consistent random seeds | 32p Protocol | LEVEL B | YES as protocol | Seed values not located; do not invent |
| Threshold selection | Validation-selected; primary results on test set | 32p Protocol | LEVEL B | YES procedure only | Numeric threshold / metric / tie-break UNKNOWN |
| Primary GT semantics | Per-file labels not located | C05 audit P0-02 | UNKNOWN | Neutral phrase only | Use “reference labels used for the primary anomaly-discrimination evaluation” |
| Primary image/anomaly score | Fusion/maps described; image-level aggregation not pinned | C05 definition audit | UNKNOWN | Do not claim max-valid recipe for 0.894 | Current Layer B recipe is v1_1, not C05 |
| **Primary fusion mode** | 32p describes entropy fusion as the system; current core is fixed-weight; C05 unreproduced | Ledger C02/C05 | **UNRESOLVED (C)** | **Must not attach entropy or fixed mix to 0.894** | Do not infer from v1_1 |
| Human-reviewed fusion | Fixed-coefficient Layer B; AUROC 0.772 | IND_EVAL_V1_1 measured | LEVEL A | YES | Scope fixed mix to this analysis only |
| PaDiM/PatchCore headline | AUROC 0.856 | Abstract + 32p Table | LEVEL B | YES | Combined column; PaDiM-only vs PatchCore-only UNKNOWN |
| WRN-50-2 backbone | 32p table header; current `wide_resnet50_2` default | 32p + padim.py/patchcore.py | LEVEL B | YES backbone + same-split policy | Do not claim local `.pth` banks produced 0.856 |
| Primary vs Layer C | 0.894/0.847/0.823 are detection metrics | 32p Results | LEVEL B | YES | Not curiosity/diversity/buffer scores |
| Lightweight FPS | 28.1 @ 256×256; excludes learned DPT/AE | Abstract + IAC hardware scope | LEVEL B | YES | No Jetson/full-pipeline timing |
| Random seed values | “consistent random seeds” only | 32p Protocol | UNKNOWN values | Protocol wording only | Do not invent a seed |
| Abstract | Entropy fusion in system description; 0.894 metrics | Accepted abstract / `main.tex` | Keep | **No abstract edit** | Body must not claim 0.894 used fixed coefficients |

## Fusion-mode decision

**Outcome C (UNRESOLVED).** C05 remains `accepted_abstract_reproduction_pending`. The current fixed-coefficient mix is the human-reviewed (v1_1) path and must not be inferred onto 0.894. The 32p manuscript describes entropy-weighted fusion as the ARTPS method and reports 0.894 without a pinned fusion config hash. Attaching either mode to the primary result would be a guess.

## Abstract

No change. The accepted abstract describes the ARTPS system (including entropy-weighted fusion) and reports 0.894. The full-paper contradiction is the body phrase “the reported evaluation uses fixed-coefficient,” which incorrectly covers the primary result. Fix the body.
