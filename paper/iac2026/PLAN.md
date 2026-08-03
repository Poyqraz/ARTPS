# IAC 2026 manuscript plan (section-by-section)

LaTeX master: [`main.tex`](main.tex) + [`iac2026.sty`](iac2026.sty). Word template is visual reference only (`IAC 2026_manuscript_template.doc`). No official IAF `.cls`.

- **Paper code:** `IAC-26,A3,IP,109,x109221` (canonical in `main.tex`)
- **Congress:** 77th IAC 2026, Antalya, Türkiye, 5–9 October 2026 (header in `iac2026.sty`)

Claim binding: [`CLAIM_EVIDENCE_LEDGER.md`](CLAIM_EVIDENCE_LEDGER.md). Experiments: [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md). Figures/tables: [`FIGURE_TABLE_PLAN.md`](FIGURE_TABLE_PLAN.md). Safety: [`SAFETY_CASE_MATRIX.md`](SAFETY_CASE_MATRIX.md). Refs: [`REFERENCE_CLEANUP.md`](REFERENCE_CLEANUP.md).

Source full MS: `docs/Full_Baydemir_ARTPS.pdf` (32p) — **compress/rewrite**, do not dump.

## Section outline

1. **Introduction** (`sections/introduction.tex`) — problem, gap, contributions; Related Work folded in.
2. **Material and Methods** (`sections/methods.tex`) — architecture; depth-guided enhancement; cues; entropy fusion; localization + FP/shadow + size–distance; curiosity; diversity; Priority Buffer; Real-ESRGAN one paragraph.
3. **Experimental Protocol** (`sections/experiments.tex`) — dataset, labeling, split, baselines, metrics (names), hardware, ablation/proxy, Jetson protocol pointer.
4. **Results** (`sections/results.tex`) — stubs only; accepted-abstract numbers are `accepted_abstract_reproduction_pending`; no rewritten tables in this milestone.
5. **Discussion** (`sections/discussion.tex`) — implemented vs proxy vs planned; failures; relative-depth limit; safety-aware / not flight-qualified wording.
6. **Conclusions** (`sections/conclusion.tex`) — short; no industrial laundry list.
7. **Declaration of Generative AI Use** (`sections/declaration.tex`) — language verification / grammar / readability only; science is author-produced and verified. Move if official IAC template requires another slot.

## Abstract policy

- Keep accepted numbers: 0.894 AUROC, 0.847 AUPRC, 0.823 F1, baseline 0.856 AUROC, 28.1 FPS @ 256×256 (lightweight core, no learned depth/AE).
- Ledger status for those numbers: `accepted_abstract_reproduction_pending`.
- **Do not** put preliminary proxy ablation sentences in the abstract (n=21 curated; no human bbox GT).

## Submission checklist (format / disclosure)

- `CORRESPONDING_EMAIL_TBD` is OK for the planning milestone only; camera-ready / submission-ready PDFs must replace it with the author-supplied address (never invent or scrape GitHub). Strict: `python _check_submission_ready.py`; planning: add `--allow-email-placeholder`.
- Visually compare PDF header (full-width two-line congress + copyright) to the official IAC Word template (CI artifact `iac2026-manuscript-preview`).
- Confirm AI disclosure remains language-only (author-defined official boundary in `sections/declaration.tex`).
- GitHub Actions validates LaTeX build, US Letter size, forbid-strings, and planning submission-ready checks under `paper/iac2026/`.
- `paper/iac2026` remains the only canonical manuscript workspace.

## Repo features → sections

| Feature | Section | Evidence |
|---------|---------|----------|
| Rover + boundary-shadow FP | Methods + Results (preliminary proxy) | C09 `proxy` / mechanism `implemented` |
| Object-in-shadow gate | Methods + Results (preliminary proxy) | C10 |
| Size–distance policy | Methods + Results (preliminary proxy) | C11–C12 |
| Tight box merge | Methods | `implemented` |
| Depth-on-RGB QC | Methods | C16 `implemented` |
| Priority Buffer + diversity | Methods + Discussion | C03–C04 `implemented` |
| Real-ESRGAN | Methods one paragraph | C13 `unsupported` |

## Cut / keep (from 32p)

**Cut:** TOC; ViT/FFN derivations; textbook P/R formulas; Installation; long UI; industrial applications list; unused BERT/LSTM/GAN refs.

**Keep equations/rules:** depth-guided enhancement; entropy fusion; curiosity score; soft similarity; uncertainty/disagreement; Priority Buffer; size–distance policy.
