# IAC 2026 LaTeX manuscript (skeleton)

Master document for the IAC 2026 full manuscript. Official IAF currently publishes a **Word** manuscript template and PDF guidelines — **no official LaTeX class**. Do **not** invent or download an unverified `IAC.cls`.

| Role | Path |
|------|------|
| Scope / cut-keep / old→new map | [SCOPE.md](SCOPE.md) |
| Style (article helper) | [iac2026.sty](iac2026.sty) |
| Manuscript | [main.tex](main.tex) |
| Bibliography | [refs.bib](refs.bib) |
| Visual reference | [`IAC 2026_manuscript_template.doc`](../../IAC%202026_manuscript_template.doc) (repo root) |
| 32-page source (do not dump) | [`Full_Baydemir_ARTPS.pdf`](../Full_Baydemir_ARTPS.pdf) |

## Build

From this directory (MiKTeX / TeX Live):

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or from repo root:

```text
pdflatex -output-directory=docs/iac2026 docs/iac2026/main.tex
```

(`bibtex` must run with `docs/iac2026` as cwd so `refs.bib` resolves.)

## Checklist vs Word template

- US Letter, ~25 mm margins, Times 10 pt, ~6 mm column gutter
- Title + abstract single-column block; body two-column
- Header: congress + copyright; footer: paper code + Page X of Y
- Abstract: third person, one paragraph, ≤400 words
- ≤6 keywords
- Numbered citations in order of appearance
- No table of contents

## Depth language

Use relative depth ordering / apparent size / image-relative near–far only. See `results/paper_figs/depth_semantics.md`.
