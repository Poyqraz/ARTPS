# IAC 2026 manuscript scope

Source full manuscript: [`../Full_Baydemir_ARTPS.pdf`](../Full_Baydemir_ARTPS.pdf) (32 pages). **Do not paste it into the IAC template.** Compress into the six sections in [`main.tex`](main.tex).

Visual/format reference (not a LaTeX class): [`../../IAC 2026_manuscript_template.doc`](../../IAC%202026_manuscript_template.doc). LaTeX is the master document (`article` + [`iac2026.sty`](iac2026.sty)).

Depth wording: relative depth ordering / apparent size / image-relative near–far — no metric distance claims. See [`../../results/paper_figs/depth_semantics.md`](../../results/paper_figs/depth_semantics.md).

## Repo features → sections

| Feature | Section | Evidence |
|---------|---------|----------|
| Rover-body + boundary-shadow FP suppression | Material and Methods + Results | `src/core/false_positive_masks.py`; `results/paper_figs/iac_shadow_proxy_*` |
| Object-in-shadow protection gate | Material and Methods + Results | gated suppression; shadow FPR / rock-loss proxy |
| Size–distance policy | Material and Methods + Results | `src/core/size_distance.py`; `iac_size_distance_proxy_*` |
| Tightened box merge | Material and Methods (+ Results) | merge/keep gates in `app.py` |
| Depth-on-RGB visualization QC | Material and Methods | depth viz quality checks |
| Priority Buffer + diversity | Material and Methods + Discussion | operational target policy |
| Real-ESRGAN | Material and Methods — **one paragraph only** | optional enhance; **no ablation → not in Results** |

Proxy evals are class/mask/OFF-run pseudo-GT, **not** human bbox GT. Lite size/distance bench = software verification only.

## Target outline (6 sections)

1. **Introduction** — problem, literature gap, contributions; Related Work folded in (short subsection OK).
2. **Material and Methods** — architecture; depth-guided enhancement; anomaly cues; entropy fusion; localization (merge, FP/shadow, size–distance); curiosity score; diversity; Priority Buffer; ESRGAN one paragraph.
3. **Experimental Protocol** — dataset, labeling, split, baselines, metrics (names only), hardware, ablation / proxy protocol.
4. **Results** — detection, ranking, localization, field-condition, hardware; shadow + size–distance proxy tables.
5. **Discussion** — interpretation, failure cases, relative-depth limit, operational safety, flight-readiness.
6. **Conclusions** — short; no industrial laundry list.

## Old § → new

| Full PDF (32p) | IAC compact |
|----------------|-------------|
| Contents | **Drop** |
| §1 Intro + §2 Related Work | Introduction |
| §3 Method | Material and Methods (compress; no ViT/FFN derivations) |
| §4 Setup + §5 Ablation | Experimental Protocol |
| §6 Results | Results |
| §7 Limitations + §8 Safety | Discussion (Priority Buffer rule also short in Methods) |
| §9 Implementation / UI / Installation | **Drop** or 1 software note paragraph |
| §10.3 Industrial applications | **Drop / severe cut** |
| §10 Conclusions | Conclusions |
| §11 References | Pruned bib |

## Cut / severely shorten

- Table of contents
- ViT attention / FFN standard formula derivations
- Precision/recall-style textbook metric formulas (name + one-line definition enough)
- Installation section
- Long UI walkthrough
- Generic industrial applications list
- Unused citations (BERT, LSTM, GAN, …)

## Keep (original equations / rules)

- Depth-guided enhancement
- Entropy-weighted fusion
- Curiosity score
- Soft similarity penalty
- Uncertainty / disagreement
- Priority Buffer rule
- Size–distance policy

## This PR vs later

This folder ships **skeleton + SCOPE + style**, not a full rewrite of the 32-page prose.
