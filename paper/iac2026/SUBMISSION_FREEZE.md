# IAC 2026 submission freeze

`submission_candidate: true`

`reviewer_facing_polish_complete: true`

`enriched_fullpaper_candidate: true`

`enriched_context_pass: true`

`cue_decomposition_figure: true`

`scientific_definition_pass_complete: true`

`close_far_qualitative_figure: true`

`iac2026_template_conformance_complete: true`

`final_visual_flow_polish_complete: true`

`titleblock_affiliation_align_complete: true`

`official_usletter_geometry_complete: true`

This record is a repository freeze for camera-ready IAC manuscript readiness.
It is **not** an official IAF submission receipt and **not** an accepted full-paper notice.

## Identifiers

| Field | Value |
| --- | --- |
| Paper ID | `IAC-26,A3,IP,109,x109221` |
| Branch | `paper/iac2026-submission-freeze` |
| Base (`main` @ PR #34) | `aeedcebfa7cd83a58879f3bef81aa0eec12e66d6` |
| Freeze commit | `4419dae72b4d35c042a30bd48792c29973ac5446` |
| Build date | 2026-08-08 (pdfinfo CreationDate 23:50:58 Türkiye) |
| Page count | 7 |
| Page size | US Letter (612 × 792 pts) |

## Author metadata

- Corresponding author: Poyraz Baydemir
- Affiliation: Faculty of Technology, Selçuk University, Konya, Türkiye (author-supplied; left-aligned 10 pt italic)
- Email: `poyrazbaydemir@gmail.com` (author-supplied; replaced `CORRESPONDING_EMAIL_TBD` only)
- Authorship order unchanged; affiliation marker `a`; corresponding line left-aligned upright

## Scientific freeze snapshot

Historical accepted-abstract values (unchanged):

- AUROC `0.894`
- AUPRC `0.847`
- F1 `0.823`
- Baseline AUROC `0.856`
- Lightweight core `28.1` FPS @ `256×256`

Supplementary `independent_eval_v1_1` (unchanged):

- AUROC `0.772`
- AP `0.956`

Not reintroduced into Results headlines (audit artifacts only):

- F1 `0.920`
- threshold `0.0`
- TN/FP/FN/TP `0/8/0/46`

Test policy (unchanged):

- `test_opened: false`
- `final_test_authorized: false`
- No final-test metric

Figure policy: **Decision B** (architecture-only Fig. 1)

- No legacy Figs 2–8
- No Tables 2–6 import
- No new qualitative figure
- No new experiment

AI declaration: unchanged (language-only).
Funding / grant / sponsorship: none added.
Accepted abstract body: unchanged (`0.772` not inserted; relative depth; lightweight-core FPS).

## Bibliography

Verified cited keys (4):

- `estlin2012` — AEGIS, ACM TIST 2012, DOI `10.1145/2168752.2168764`
- `ranftl2021dpt` — DPT / ICCV 2021, DOI `10.1109/ICCV48922.2021.01196`
- `defard2020padim` — PaDiM, DOI `10.1007/978-3-030-68799-1_35`
- `roth2022patchcore` — PatchCore / CVPR 2022, DOI `10.1109/CVPR52688.2022.01392`

ISAIRAS 2014: not added (`UNVERIFIED_METADATA`).

## Hygiene

| Class | Count | Notes |
| --- | --- | --- |
| PDF_VISIBLE TODO/TBD/PLACEHOLDER/FIXME/`CORRESPONDING_EMAIL_TBD` | 0 | 7-page PDF text scan |
| SOURCE_COMMENT_ONLY | 6 | `% TODO` in methods (3), experiments, discussion, declaration |
| PLANNING_DOC_ONLY | n/a | Historical markdown (`PLAN.md`, audits, ledger, protocol) |
| SUBMISSION_BLOCKER | 0 | |

## Checks

| Check | Result |
| --- | --- |
| Strict `_check_submission_ready.py` (no `--allow-email-placeholder`) | PASS (`email_placeholder_absent`) |
| `pytest -q tests/iac2026` | 216 passed |
| `git diff --check` | clean |
| pdflatex → bibtex → pdflatex ×2 | exit 0 |
| Overfull `\hbox` | 0 |
| Underfull `\hbox` | 32 (warn-only; not a blocker) |
| Undefined citations / references | 0 |
| Visual 7-page inspection | PASS (header/ID/email/Fig. 1 A–B–C/tables/refs) |

## CI

- IAC paper build: PASS ([run 31278095264](https://github.com/Poyqraz/ARTPS/actions/runs/31278095264))
- IAC reproduction harness: PASS ([run 31278095266](https://github.com/Poyqraz/ARTPS/actions/runs/31278095266)); triggered by `tests/iac2026/**`

## Reviewer-facing polish

- `reviewer_facing_polish_complete: true`
- polish_commit: `8f07a9432ee36c22ba2936bc7ed91c7cdf9e3cea`
- Scientific freeze snapshot above unchanged (metrics, Decision B, test flags, `submission_candidate`)

## Enriched full-paper candidate

- `enriched_fullpaper_candidate: true`
- 7-page freeze manuscript remains the fallback on `main`
- See [`ENRICHED_FULLPAPER.md`](ENRICHED_FULLPAPER.md) and [`figures/FIG_QUALITATIVE_PROVENANCE.md`](figures/FIG_QUALITATIVE_PROVENANCE.md)

## Enriched context pass

- `enriched_context_pass: true`
- Evaluation-track table + operational suppression / ranking context; scientific snapshot unchanged

## Cue-decomposition figure

- `cue_decomposition_figure: true`
- Same non-test Fig.~2 sample; frozen fusion cues only; no new metric, no test split

## Scientific definition pass

- `scientific_definition_pass_complete: true`
- Fig.~1 topology updated (classifier to candidate scoring; proximity; candidate score)
- Fig.~2(b) post-suppression combined map; Fig.~3(d) pre-suppression fused map + independent min-max caveat
- Frozen map equation and candidate/image score documented
- Experimental protocol finalized; planning/draft tone removed
- Literature: Francis 2017, Francis 2019, Gaines 2020 added
- Historical and supplementary metrics unchanged; test unopened; no new quantitative experiment
- definition_pass_commit: `4cda8f6`
- pages: 11 (US Letter); overfull `\hbox`: 0; underfull `\vbox`: 22 (warn-only)
- local verify: `pytest -q tests/iac2026` 220 passed; `_check_submission_ready.py` strict OK; `git diff --check` clean; undefined cites/refs: 0
- CI: IAC paper build PASS ([run 31312030326](https://github.com/Poyqraz/ARTPS/actions/runs/31312030326)); harness PASS ([run 31312030318](https://github.com/Poyqraz/ARTPS/actions/runs/31312030318))
- test unopened; historical and supplementary metrics unchanged; no new quantitative experiment

## Close vs distant qualitative figure

- `close_far_qualitative_figure: true`
- Author-provided non-test pool only; Fig.~4 illustrative close/far pair; no new metric, no test split

## Template conformance pass

- `iac2026_template_conformance_complete: true`
- Official: IAC 2026 Manuscript Guidelines (IAF PDF)
- Community reference only: grande-dev `IAC_style.cls` (2024-based / 2025 update); not adopted wholesale; no Roman+underline headings; copyright Form A / IAF-held unchanged
- Geometry: left/right 2.25 cm, top/bottom 3.35 cm, `columnsep` 0.8 cm; US Letter 612×792
- Font: `newtxtext` + `newtxmath`, 10 pt T1 (embedded TeXGyreTermes / NewTX)
- Fig. caption naming: `Fig.`; in-text `Fig.~\ref`; tables remain `Table` above
- Fig. 1: B heading / entropy text overlap fixed (bbox intersection area 0); A/B/C topology unchanged
- Raster: 180 dpi + PNG optimize; Fig. 2/3/4 sources, xywh, scores unchanged
- Pages: 11; PDF bytes: 4,060,416 (≤4.5 MB target; CI hard &lt;5 MB)
- Metrics unchanged (historical 0.894/0.847/0.823/0.856/28.1; supp 0.772/0.956)
- `test_opened: false`; `final_test_authorized: false`
- Paper code: `IAC-26,A3,IP,109,x109221` (title block + footer)
- local verify: `pytest -q tests/iac2026` 232 passed; `_check_submission_ready.py` strict OK; `git diff --check` clean; overfull `\hbox` 0

## Final visual-flow polish

- `final_visual_flow_polish_complete: true`
- Subsection headings ragged-right; 4.2/4.3/5.2 titles shortened; protocol IDs retained in first body sentence
- Table 1 after complete 4.4 paragraph: no mid-sentence split; still `table*`
- Table 3 caption ragged, no monospaced protocol ID; captions remain above
- Fig. 2 after 3.3; Fig. 3 after 3.4; Fig. 4 after qualitative subsection (`\FloatBarrier`); order Fig. 1–4 unchanged
- Fig. 4 canvas `(7.4, 4.83)` / 9 pt titles (~8 pt physical); overlay `candidate_support_v3` dual-stroke halo; Fig. 3 PNG unchanged
- Pages: 11; PDF bytes: 3,446,057; US Letter; overfull 0
- Metrics unchanged; `test_opened: false`; no new experiment
- local verify: `pytest -q tests/iac2026` 232 passed; `_check_submission_ready.py` strict OK; `git diff --check` clean

## Title-block affiliation alignment

- `titleblock_affiliation_align_complete: true`
- Centered: paper code, title (`\large\bfseries`), author (`\normalsize\bfseries`, `$^{a,*}$`)
- Left-aligned: `$^{a}$ Faculty of Technology, Selçuk University, Konya, Türkiye` (`\normalsize\itshape`)
- Left-aligned: `* Corresponding author: poyrazbaydemir@gmail.com` (`\normalsize\upshape`)
- Header/footer wording and rules unchanged; paper code `IAC-26,A3,IP,109,x109221`
- Abstract/keywords/metrics unchanged; `\setstretch{1.0}`; `\parskip=0pt`; `\parindent=12pt`
- Spacing variant B kept: author→affil 0.25em, affil→corresponding 0.35em, corresponding→Abstract 0.6em
- Pages: 11; PDF bytes: 3,446,871; US Letter; overfull 0
- local verify: `pytest -q tests/iac2026` 232 passed; `_check_submission_ready.py` strict OK; `git diff --check` clean

## Official US-Letter geometry (Word/PDF template)

- `official_usletter_geometry_complete: true`
- Oracle: `paper/template/IAC 2026_manuscript_template (2).pdf` (SHA256 `27137db38af98a26cc879287b0c2914985260fffec7d1e492e92311782db888f`); PyMuPDF coords
- Geometry: `left=0.98in`, `right=0.95in`, `top=0.99in`, `bottom=1.0in`, `columnsep=0.22in`
- Header: 8 pt fancyhdr; `5--9 Oct 2026`; Form A copyright `\textcopyright2026`
- Title: `\normalsize\bfseries` (10 pt bold; matches official template)
- Footer: 10 pt; centered `\makebox[5.58in]`; no footnotesize
- Measured ARTPS ≈ official within ±0.03 in (column gap ±0.02 in); grade A
- Pages: 11; PDF bytes: 3,445,862; US Letter; overfull 0; science unchanged
- local verify: `pytest -q tests/iac2026` 232 passed; `_check_submission_ready.py` strict OK
