# IAC 2026 manuscript plan (section-by-section)

LaTeX master: [`main.tex`](main.tex) + [`iac2026.sty`](iac2026.sty). Word template is visual reference only (`IAC 2026_manuscript_template.doc`). No official IAF `.cls`.

- **Paper code:** `IAC-26,A3,IP,109,x109221` (canonical in `main.tex`)
- **Congress:** 77th IAC 2026, Antalya, Türkiye, 5–9 October 2026 (header in `iac2026.sty`)
- **Full-paper v0.1:** prose-first draft (Introduction–Conclusion + Limitations); Results evidence-gated
- **Sprint:** [`FULL_PAPER_SPRINT_PLAN.md`](FULL_PAPER_SPRINT_PLAN.md) (Aug 5–Sep 14; **no funding/grant work**)

Claim binding: [`CLAIM_EVIDENCE_LEDGER.md`](CLAIM_EVIDENCE_LEDGER.md). Experiments: [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md). Figures/tables: [`FIGURE_TABLE_PLAN.md`](FIGURE_TABLE_PLAN.md). Safety: [`SAFETY_CASE_MATRIX.md`](SAFETY_CASE_MATRIX.md). Refs: [`REFERENCE_CLEANUP.md`](REFERENCE_CLEANUP.md).

Source full MS: `docs/Full_Baydemir_ARTPS.pdf` (32p) — **compress/rewrite**, do not dump.

## Section outline (full-paper v0.1)

1. **Introduction** (`sections/introduction.tex`) — problem, gap, ARTPS, six defendable contributions (no first/SOTA).
2. **Related Work** (`sections/related_work.tex`) — short 2.1–2.4; thin verified cites + TODO(citation).
3. **ARTPS Methodology** (`sections/methods.tex`) — subsections through Priority Buffer + fail-closed; pipeline `figure*` placeholder; primary score `max_valid_candidate_after_masks`; no metric-depth claims.
4. **Experimental Protocol** (`sections/experiments.tex`) — historical accepted-abstract vs `independent_eval_v1`; annotation/dataset plan language; baselines; SHA/fail-closed; bootstrap planned not fabricated.
5. **Results** (`sections/results.tex`) — Table 1 accepted-abstract reference + footnote; independent_eval pending prose; proxy note only (no invented numbers).
6. **Discussion** (`sections/discussion.tex`) — method-focused; `% TODO(results):` hooks; no outperforms/demonstrate.
7. **Limitations** (`sections/limitations.tex`) — relative depth; pending reproduction/data; not flight-qualified; bounded safety.
8. **Conclusions** (`sections/conclusion.tex`) — architecture vs accepted-abstract vs pending independent eval vs future HW.
9. **Declaration of Generative AI Use** (`sections/declaration.tex`) — **frozen** language verification only; do not edit casually.

## Claim-status freezes

| ID | Support (do not change casually) |
|----|----------------------------------|
| C05 / C06 / C07 | `accepted_abstract_reproduction_pending` |
| IND_EVAL_V1 | `protocol_defined_pending_data` |
| C08 | `planned` |

## Abstract policy

- Keep accepted numbers: 0.894 AUROC, 0.847 AUPRC, 0.823 F1, baseline 0.856 AUROC, 28.1 FPS @ 256×256 (lightweight core, no learned depth/AE).
- Ledger status for those numbers: `accepted_abstract_reproduction_pending`.
- **Do not** put preliminary proxy ablation sentences in the abstract.
- **Do not** add funding / grant / sponsorship / budget language anywhere.

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
| Real-ESRGAN | Methods one paragraph (optional later) | C13 `unsupported` |
| Independent eval v1 | Protocol + Results stub | IND_EVAL_V1 pending data |

## Cut / keep (from 32p)

**Cut:** TOC; ViT/FFN derivations; textbook P/R formulas; Installation; long UI; industrial applications list; unused BERT/LSTM/GAN refs; funding prose.

**Keep equations/rules:** depth-guided enhancement; entropy fusion; curiosity score; soft similarity; uncertainty/disagreement; Priority Buffer; size–distance policy.
