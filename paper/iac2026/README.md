# IAC 2026 manuscript + evidence plan

This folder is the **claim–evidence planning milestone** for IAC 2026 (PR framing: establish manuscript and evidence plan). It is **not** a full prose rewrite of `docs/Full_Baydemir_ARTPS.pdf`.

| File | Role |
|------|------|
| [PLAN.md](PLAN.md) | Section-by-section outline, cut/keep |
| [CLAIM_EVIDENCE_LEDGER.md](CLAIM_EVIDENCE_LEDGER.md) | Abstract claim ↔ experiment ↔ script ↔ table ↔ support |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | Dataset, annotation, ablation, proxy, Jetson protocol |
| [FIGURE_TABLE_PLAN.md](FIGURE_TABLE_PLAN.md) | Figure/table inventory + experiment map |
| [REFERENCE_CLEANUP.md](REFERENCE_CLEANUP.md) | Bib keep/drop |
| [SAFETY_CASE_MATRIX.md](SAFETY_CASE_MATRIX.md) | Operational safety ↔ evidence |
| [main.tex](main.tex) + [sections/](sections/) | Compilable stubs |
| [iac2026.sty](iac2026.sty) | Informal layout (`article`); Word template = visual ref |
| [references.bib](references.bib) | Minimal bib |

Official IAF publishes Word + PDF guidelines — **no official LaTeX class**. Do not invent `IAC.cls`.

## Build

```text
cd paper/iac2026
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
python _count_abstract.py
```

## Rules

- No Results number rewrite until ledger rows close.
- No metric distance / metric 3D size claims.
- `src/ui/i18n/tr.py` is app localization; not part of this English manuscript pack.
