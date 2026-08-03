# IAC 2026 manuscript plan (section-by-section)

LaTeX master: [`main.tex`](main.tex) + [`iac2026.sty`](iac2026.sty). Word template is visual reference only (`IAC 2026_manuscript_template.doc`). No official IAF `.cls`.

Claim binding: [`CLAIM_EVIDENCE_LEDGER.md`](CLAIM_EVIDENCE_LEDGER.md). Experiments: [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md). Figures/tables: [`FIGURE_TABLE_PLAN.md`](FIGURE_TABLE_PLAN.md). Safety: [`SAFETY_CASE_MATRIX.md`](SAFETY_CASE_MATRIX.md). Refs: [`REFERENCE_CLEANUP.md`](REFERENCE_CLEANUP.md).

Source full MS: `docs/Full_Baydemir_ARTPS.pdf` (32p) — **compress/rewrite**, do not dump.

## Section outline

1. **Introduction** (`sections/introduction.tex`) — problem, gap, contributions; Related Work folded in.
2. **Material and Methods** (`sections/methods.tex`) — architecture; depth-guided enhancement; cues; entropy fusion; localization + FP/shadow + size–distance; curiosity; diversity; Priority Buffer; Real-ESRGAN one paragraph.
3. **Experimental Protocol** (`sections/experiments.tex`) — dataset, labeling, split, baselines, metrics (names), hardware, ablation/proxy, Jetson protocol pointer.
4. **Results** (`sections/results.tex`) — stubs only until ledger rows close; no rewritten numbers in this milestone.
5. **Discussion** (`sections/discussion.tex`) — measured vs proxy; failures; relative-depth limit; safety; flight-readiness wording.
6. **Conclusions** (`sections/conclusion.tex`) — short; no industrial laundry list.

## Repo features → sections

| Feature | Section | Evidence |
|---------|---------|----------|
| Rover + boundary-shadow FP | Methods + Results (proxy) | C09 |
| Object-in-shadow gate | Methods + Results (proxy) | C10 |
| Size–distance policy | Methods + Results (proxy) | C11–C12 |
| Tight box merge | Methods | code |
| Depth-on-RGB QC | Methods | C16 |
| Priority Buffer + diversity | Methods + Discussion | C03–C04 |
| Real-ESRGAN | Methods one paragraph | C13 unsupported |

## Cut / keep (from 32p)

**Cut:** TOC; ViT/FFN derivations; textbook P/R formulas; Installation; long UI; industrial applications list; unused BERT/LSTM/GAN refs.

**Keep equations/rules:** depth-guided enhancement; entropy fusion; curiosity score; soft similarity; uncertainty/disagreement; Priority Buffer; size–distance policy.
