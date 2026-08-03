# IAC 2026 manuscript + evidence plan

This folder is the **claim–evidence planning milestone** for IAC 2026 (paper code **IAC-26,A3,IP,109,x109221**). It is **not** a full prose rewrite of `docs/Full_Baydemir_ARTPS.pdf`.

Congress: 77th International Astronautical Congress (IAC 2026), Antalya, Türkiye, 5–9 Oct 2026.

| File | Role |
|------|------|
| [PLAN.md](PLAN.md) | Section-by-section outline, cut/keep |
| [CLAIM_EVIDENCE_LEDGER.md](CLAIM_EVIDENCE_LEDGER.md) | Abstract claim ↔ experiment ↔ script ↔ table ↔ support |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | Dataset, annotation, ablation, proxy, Jetson protocol |
| [FIGURE_TABLE_PLAN.md](FIGURE_TABLE_PLAN.md) | Figure/table inventory + experiment map |
| [REFERENCE_CLEANUP.md](REFERENCE_CLEANUP.md) | Bib keep/drop; camera-ready TODOs |
| [SAFETY_CASE_MATRIX.md](SAFETY_CASE_MATRIX.md) | Operational safety ↔ evidence |
| [main.tex](main.tex) + [sections/](sections/) | Compilable stubs |
| [iac2026.sty](iac2026.sty) | Informal layout (`article`); Word template = visual ref |
| [references.bib](references.bib) | **Minimal skeleton bib for builds only** (not camera-ready) |

Official IAF publishes Word + PDF guidelines — **no official LaTeX class**. Do not invent `IAC.cls`. Work only under `paper/iac2026/` (do not recreate `docs/iac2026/`).

## Support levels (short)

`implemented` · `measured` · `accepted_abstract_reproduction_pending` · `proxy` · `software_verification` · `planned` · `unsupported`

Accepted abstract numbers (0.894 AUROC / 0.847 AUPRC / 0.823 F1 / 0.856 baseline / 28.1 FPS) stay in the abstract with ledger status `accepted_abstract_reproduction_pending`. Proxy ablations are **not** in the abstract.

## Build

```text
cd paper/iac2026
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
python _count_abstract.py
```

(`_count_abstract.py` also works from the repository root.)

## Rules

- No Results number rewrite in this milestone.
- No metric distance / metric 3D size claims (disclaimer / forbidden only).
- `src/ui/i18n/tr.py` is app localization; not part of this English manuscript pack.
- Do not commit generated `.pdf` / `.aux` / `.log` / `.bbl` / `.blg`.
