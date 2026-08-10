# IAC 2026 template-conformance audit

Authority: **official IAC 2026 Manuscript Guidelines** (IAF PDF) wins.
Community [`grande-dev/International-Astronautical-Congress-LaTeX-template`](https://github.com/grande-dev/International-Astronautical-Congress-LaTeX-template) (`IAC_style.cls`, 2024-based / 2025 update) is visual/LaTeX reference only.
ARTPS keeps `article` + `iac2026.sty`; does **not** adopt `IAC_style.cls` wholesale.

Copyright: visible notice is **Form A / IAF-held**. Do **not** switch to B1/B2/C. Online upload selection must match the printed notice.

Science freeze (unchanged this pass): historical `0.894/0.847/0.823/0.856/28.1`; supplementary `0.772/0.956`; `test_opened=false`; paper code `IAC-26,A3,IP,109,x109221`.

## Baseline (`main` before this pass)

| Metric | Value |
| --- | --- |
| Pages | 12 |
| PDF bytes | 5,509,673 (~5.51 MB) |
| Page size | US Letter 612 × 792 pt |
| Overfull `\hbox` | 0 |
| Underfull `\hbox` | ~10 (warn-only) |
| Undefined cite/ref | 0 |
| Fonts | embedded NimbusRom (mathptmx) + Type3 + CM math |
| Fig. 2 PNG | 1,408,921 B @ 200 dpi |
| Fig. 3 PNG | 1,671,287 B @ 200 dpi |
| Fig. 4 PNG | 2,010,361 B @ 200 dpi |
| Raster sum | ~5.09 MB |

Preview pages / contact sheet: `paper/iac2026/_preview_pages/` (untracked).

## Requirements

| Requirement | Official 2026 rule | 2025 LaTeX reference | Current ARTPS (baseline) | Action | Final status | Tag |
| --- | --- | --- | --- | --- | --- | --- |
| Paper size | US Letter | letterpaper | 612×792 | keep | PASS | OFFICIAL_2026 |
| Margins | ~2.25 cm L/R, ~3.35 cm T/B | geometry in `IAC_style.cls` | 25 mm + `includehead`/`includefoot` | 2.25/3.35 cm, no includehead/foot unless collision | PASS (post-pass) | OFFICIAL_2026 |
| Column gutter | ~0.8 cm | `columnsep` ~0.8 cm | 6 mm | `columnsep=0.8cm` | PASS (post-pass) | OFFICIAL_2026 |
| Body font | Times 10 pt, embedded | `newtx` / Times-like | `mathptmx` 10 pt | `newtxtext`+`newtxmath`, T1 | PASS (post-pass) | OFFICIAL_2026 |
| Paper ID | on manuscript | `\IACpapernumber` above title | footer only | ID `\normalsize` above title; keep footer | PASS (post-pass) | OFFICIAL_2026 |
| Title | large bold, not oversized | `\large\bfseries` | `\Large\bfseries` | drop `\Large` → `\large\bfseries` | PASS (post-pass) | OFFICIAL_2026 |
| Author / affil | readable | bold author; italic affil | similar | `\normalsize\bfseries` author; `\small\itshape` affil; corresponding upright | PASS (post-pass) | OFFICIAL_2026 |
| Abstract heading | centered bold Abstract | centered **Abstract** | flush-left bold | centered bold Abstract | PASS (post-pass) | OFFICIAL_2026 |
| Abstract body | as submitted | n/a | 5-arg `\IACmaketitle` #4 | byte-for-byte unchanged | PASS | OFFICIAL_2026 |
| Keywords | present | Keywords line | present | keep | PASS | OFFICIAL_2026 |
| Header/footer rules | no heavy rules required | often rule-less | 0.4 pt head/foot rules | `headrulewidth`/`footrulewidth=0pt` | PASS (post-pass) | OFFICIAL_2026 |
| Header content | congress + copyright | two-line header | two-line + rules | keep wording; no rules; `headheight=32pt` | PASS (post-pass) | OFFICIAL_2026 |
| Footer | paper code + Page X of Y | left ID / right pages | same | keep | PASS | OFFICIAL_2026 |
| Copyright notice | Form A / IAF-held | community B/C variants exist | IAF-held wording | **do not** switch to B1/B2/C | PASS (unchanged) | OFFICIAL_2026 |
| Section numbering | arabic 1. / 1.1 / 1.1.1 | community also has Roman+underline variant | arabic bold/italic flush left | keep arabic; tighten spacing; **no** Roman+underline | PASS (post-pass) | COMMUNITY_2025_REFERENCE |
| Figure captions | Fig. n, hanging | `name=Fig.` | `Figure` default | `\captionsetup[figure]{name=Fig.,format=hang}` | PASS (post-pass) | OFFICIAL_2026 |
| In-text figure refs | Fig. n acceptable | Fig.~ | `Figure~\ref` | formatting-only → `Fig.~\ref` | PASS (post-pass) | OFFICIAL_2026 |
| Table captions | Table n above table | Table above | Table above | keep | PASS | OFFICIAL_2026 |
| Wide figures | two-column when illegible in one | `figure*` | Fig. 1–4 `\figure*`/`\textwidth` | keep; no `[H]` | PASS | OFFICIAL_2026 |
| Fig. 1–4 two-column justification | readable at print scale | n/a | already full-width | pipeline + 4/4/6 panels illegible in one column | PASS | OFFICIAL_2026 |
| Page count | typically ≤15; camera-ready often 10–14 | n/a | 12 | 12–14 acceptable; hard max 15; do not cut science | PASS (post-pass) | OFFICIAL_2026 |
| PDF size | ≤5 MB official; target ≤4.5 MB | n/a | 5.51 MB | 180 dpi + PNG optimize + pdf compress | PASS (post-pass; see bytes below) | OFFICIAL_2026 |
| Font files in repo | not required | n/a | none | none; CI `texlive-fonts-extra` | PASS | OFFICIAL_2026 |
| Tagged PDF / PDF/A project | not required for IAC upload | n/a | none | no tagged-PDF project | PASS | OFFICIAL_2026 |
| Fig. 1 overlap | figures must be readable | n/a | B heading y=4.65 vs entropy (2.45,4.55) | heading 4.90 / entropy 4.35; topology unchanged | PASS (post-pass) | OFFICIAL_2026 |
| Fig. 2/3/4 science | n/a (integrity) | n/a | locked sources/xywh/scores | typography+DPI only; fixture lock | PASS (post-pass) | OFFICIAL_2026 |

## Fig. 1–4 two-column justification

| Figure | Why one column is illegible |
| --- | --- |
| Fig. 1 | Full A/B/C pipeline (~14 nodes + dashed specified-fusion + classifier→scoring). Single column (~8 cm) collapses labels. |
| Fig. 2 | 2×2 RGB / map / depth / overlay; panel titles + heatmaps unreadable at ~8 cm. |
| Fig. 3 | 2×2 cue maps on the same source; same scale constraint as Fig. 2. |
| Fig. 4 | 2×3 close/far RGB–map–overlay; six panels require `\textwidth`. |

## PDF size (before → after)

| Item | Before (bytes) | After (bytes) | Δ % |
| --- | --- | --- | --- |
| `main.pdf` | 5,509,673 (12 pp) | 4,060,416 (11 pp) | −26.3% |
| Fig. 2 PNG | 1,408,921 | 1,037,486 | −26.4% |
| Fig. 3 PNG | 1,671,287 | 1,290,916 | −22.8% |
| Fig. 4 PNG | 2,010,361 | 1,489,835 | −25.9% |

Hard CI gate: PDF `< 5,000,000` bytes. Target: `≤ 4,500,000`. **Met** (3.87 MB). Page count 11 is below the 12–14 expected band only because geometry/heading tighten; science prose was not cut. Official hard max remains 15.

Post-pass fonts: TeXGyreTermes / NewTX (embedded Type 1); no Type3; US Letter 612×792; overfull `\hbox` 0; underfull warn-only; hyperref title/author set; tagged PDF no.

## Community class: what we did **not** copy

- Roman (I, II, …) + underline section headings
- Copyright forms B1 / B2 / C
- Wholesale `IAC_style.cls` (we keep `article` + helper sty)
