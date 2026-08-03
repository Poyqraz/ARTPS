# IAC 2026 manuscript + evidence plan

This folder is the **sole canonical manuscript workspace** for IAC 2026 (paper code **IAC-26,A3,IP,109,x109221**). It is **not** a full prose rewrite of `docs/Full_Baydemir_ARTPS.pdf`. Do not recreate `docs/iac2026/`.

Congress: 77th International Astronautical Congress (IAC 2026), Antalya, Türkiye, 5–9 October 2026.

| File | Role |
|------|------|
| [PLAN.md](PLAN.md) | Section-by-section outline, cut/keep |
| [CLAIM_EVIDENCE_LEDGER.md](CLAIM_EVIDENCE_LEDGER.md) | Abstract claim ↔ experiment ↔ script ↔ table ↔ support |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | Dataset, annotation, ablation, proxy, Jetson protocol |
| [FIGURE_TABLE_PLAN.md](FIGURE_TABLE_PLAN.md) | Figure/table inventory + experiment map |
| [REFERENCE_CLEANUP.md](REFERENCE_CLEANUP.md) | Bib keep/drop; camera-ready TODOs |
| [SAFETY_CASE_MATRIX.md](SAFETY_CASE_MATRIX.md) | Operational safety ↔ evidence |
| [main.tex](main.tex) + [sections/](sections/) | Compilable stubs (+ AI disclosure) |
| [iac2026.sty](iac2026.sty) | Informal layout (`article`); Word template = visual ref |
| [references.bib](references.bib) | **Minimal skeleton bib for builds only** (not camera-ready) |
| [`_count_abstract.py`](_count_abstract.py) | Abstract ≤400 words / ≤6 keywords |
| [`_check_submission_ready.py`](_check_submission_ready.py) | Submission-readiness validator |

Official IAF publishes Word + PDF guidelines — **no official LaTeX class**. Do not invent `IAC.cls`.

## Submission disclosures (check before camera-ready)

- **Generative AI (author-defined official boundary):** tools were used **only** for language verification, grammar checking, and readability improvement. Scientific content, methods, code, experiments, results, interpretations, and conclusions are author-produced and author-verified. See `sections/declaration.tex` (do not expand that wording casually). If the official IAC Word template requires a different declaration slot, move the text there (TODO).
- **Corresponding author email:**
  - `CORRESPONDING_EMAIL_TBD` is **temporarily allowed** for this planning milestone.
  - A submission-ready / camera-ready PDF **must not** ship with this placeholder.
  - Strict check: `python _check_submission_ready.py` (fails while placeholder remains).
  - Planning CI: `python _check_submission_ready.py --allow-email-placeholder`.
  - Do **not** invent an address or guess from a GitHub profile; the author supplies the real email manually.
- **Header:** visually compare the two-line centered PDF header against the official IAC Word manuscript template (CI uploads `main.pdf` + `main.log` as artifact `iac2026-manuscript-preview`).
- **CI:** `.github/workflows/iac-paper.yml` validates LaTeX build, US Letter page size (~612×792 pts), abstract limits, planning submission-ready checks, and formatting forbid-strings.

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
python _check_submission_ready.py --allow-email-placeholder
```

(`_count_abstract.py` / `_check_submission_ready.py` also work from the repository root.)

## Rules

- No Results number rewrite in this milestone.
- No metric distance / metric 3D size claims (disclaimer / forbidden only).
- `src/ui/i18n/tr.py` is app localization; not part of this English manuscript pack.
- Do not commit generated `.pdf` / `.aux` / `.log` / `.bbl` / `.blg`.
