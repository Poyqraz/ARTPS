# IAC 2026 full-manuscript audit (post PR #31)

Base: `main` @ `bbbb48e` (PR #31 merged).
Compiled snapshot: `paper/iac2026/main.pdf` — **6 pages**, US Letter, two-column, paper ID `IAC-26,A3,IP,109,x109221`.
Mode: research-first. This file predates TeX patches in the same PR.

Evidence layers (do not conflate):

- **A** Historical accepted / 32p manuscript: 0.894 / 0.847 / 0.823 / 0.856 / 28.1; C05–C07 `accepted_abstract_reproduction_pending`
- **B** Current repository implementation (`src/`, `app.py`)
- **C** Supplementary validation `independent_eval_v1_1`: AUROC 0.772, AP 0.956 (prevalence 46/54 ≈ 0.852); F1 0.920 / thr 0.0 / 0/8/0/46 remain **audit artifacts only**

---

## 1. Manuscript map

| Section | Approx. | Purpose | Main claims | Evidence | Duplication / gaps / transitions |
|---------|---------|---------|-------------|----------|----------------------------------|
| Title + abstract (`main.tex`) | ~1 col | Accepted IAC framing | Hybrid RGB+depth; entropy fusion; curiosity + soft similarity; Priority Buffer; historical metrics; 28.1 lightweight | Layer A (accepted abstract) | Strong words: *reliable*, *robust*, *Operationally Safe*. Body bounds safety. **Do not rewrite abstract numbers.** |
| 1 Introduction | ~0.7 p | Problem + ARTPS + 6 contributions | Relative depth only; not flight-qualified; independent_eval distinct from historical | A + B architectural | Safety paragraph repeats later. Contribution 2 states entropy fusion as if live. Contribution 6 names v1 not v1_1. |
| 2 Related Work | ~0.4 p | AEGIS-style ops; PaDiM/PatchCore; DPT; edge safety | Anomaly ≠ science target; no metric depth | Thin cites (`estlin2012`, Defard, Roth, Ranftl) | `% TODO(citation)` Estlin 2014 + edge survey. “Strong industrial baselines” slightly promotional. |
| 3 Methods | ~1.5 p | Architecture spec + Fig 1 placeholder | `max_valid_candidate_after_masks`; masks; size–distance; diversity; buffer; fail-closed | B mixed with A specification | **Zero equations.** Entropy described as used. Curiosity / similarity formulas absent vs abstract. Frozen eval path not distinguished from UI ranking stack. |
| 4 Experimental Protocol | ~1.0 p | Separate A vs v1 vs v1_1 | Historical pending; v1 test-closed; v1_1 repeat-author; no Jetson; proxies non-headline | Ledger + freeze + v1_1 reports | Metrics list still names F1 for “current track”. Transition to Results is clear. |
| 5 Results | ~0.8 p | Table 1 historical; Table 2s supplementary | 0.894…; 0.772 / 0.956; all-positive F1 not a performance measure | A + C | No 0.920 / 0.0 / confusion in table (good). Missing: frozen score = image-level max after masks, **not** curiosity/diversity/buffer. |
| 6 Discussion | ~0.6 p | Interpretation | 340/20 Mastcam-specific; AUROC 0.772 ranking; not historical GT failure | C + A separation | 340/20 stated twice. Weak on *why* hybrid/diversity/buffer. Safety paragraph repeats Limitations. |
| 7 Limitations | ~0.4 p | Compact bounds | Relative depth; reproduction pending; repeat-author; 20/8/2; thr unstable; no Jetson; not flight-qualified | C + freeze | Largely complete. Missing: frozen eval omits ranking stack. |
| 8 Conclusions | ~0.2 p | Close | Architecture; historical pending; v1 test-closed; v1_1 AUROC 0.772 | A+C | Adequate. Future harder-negative set stays future tense. |
| 9 AI declaration | ~0.1 p | Frozen language-only | Grammar/readability only | Author-defined | **No inconsistency found. Do not edit.** |
| 10 References | 4 keys | Skeleton bib | All four cited | `references.bib` | `estlin2012` = “Tara and others”; no DOI. Not camera-ready (`REFERENCE_CLEANUP.md`). |

Coherence: reads as **one gated draft**, not a dump of 32p, but still a **collection of evidence patches** around Methods/Results (entropy vs code; curiosity missing; two eval tracks). Not yet a single narrative spine.

---

## 2. Claim-term audit (manuscript TeX only)

| Term / locus | Class | Note |
|--------------|-------|------|
| Title *Operationally Safe* | SUPPORTED WITH QUALIFIER | Accepted title; body must keep “not flight-qualified / no universal guarantees”. Do not change title. |
| Abstract *reliable onboard prioritization* | SUPPORTED WITH QUALIFIER | Accepted abstract. Body already bounds. Do not rewrite abstract. |
| Abstract *robust localization* | SUPPORTED WITH QUALIFIER | Same. |
| Abstract *autonomous exploration* (keywords) | SUPPORTED WITH QUALIFIER | Onboard screening, not mission autonomy replacement. |
| Intro “does not assert universal safety guarantees” | SUPPORTED | Keep. |
| Methods “not asserted here as superior without evidence” | SUPPORTED | Good negation. |
| Methods “ARTPS uses an entropy-weighted dynamic fusion rule” | **OVERSTATED** | Historical/spec. Live `artps_detection_core` = **fixed weights**. Frozen eval uses that core. |
| Discussion “not unconstrained claim of superior detection” | SUPPORTED | Keep. |
| Discussion “useful ranking separation” (AUROC 0.772) | SUPPORTED WITH QUALIFIER | Ranking-only; 8 val negatives. Soften if patching Discussion. |
| Related Work “strong industrial anomaly baselines” | SUPPORTED WITH QUALIFIER | Cite Defard/Roth; do not imply C06 reproduced. |
| Related Work “peak accuracy” | OVERSTATED (mild) | Prefer ranking / discrimination. |
| *demonstrate / prove / outperform / SOTA / novel / first / guarantee / flight-ready / Jetson validated* | absent or negated | Good. |
| Metric depth / mineral ID / flight qualification / full-neural = 28.1 / independent second annotation / F1 0.920 headline | not claimed in Results | Good. Keep guards. |
| “verified” in declaration (“produced and verified by the author”) | SUPPORTED | Author verification of scientific content, not independent V&V. |

---

## 3. Methods component status (code vs prose)

| Component | Status | Manuscript must say |
|-----------|--------|---------------------|
| Input enhancement / Real-ESRGAN | OPTIONAL (default off); C13 `unsupported` | Optional; not frozen-eval. |
| DPT_Large relative depth | IMPLEMENTED (frozen: fail-closed local DPT) | Relative only. |
| Reconstruction AE | IMPLEMENTED | Cue, not mineral detector. |
| Image / depth cues | IMPLEMENTED | Fixed-weight fusion in core. |
| Classifier (known-value) | OPTIONAL / IMPLEMENTED | Not mineral ID. |
| Entropy-weighted fusion | **HISTORICAL / SPECIFIED**; current core = fixed-weight **PROXY** of multi-cue fusion | Must not read as measured live entropy rule. |
| Localization | IMPLEMENTED | OK. |
| Operational masks | IMPLEMENTED; proxy eval C09–C10 | Mechanism vs proxy evidence. |
| `max_valid_candidate_after_masks` | IMPLEMENTED (frozen eval contract) | **This** is the v1_1 score. |
| Curiosity score | IMPLEMENTED in UI (`curiosity_scorer.py`); **not** in frozen eval | Name it; separate from image_score. |
| Soft similarity + clustering | IMPLEMENTED in UI (`app.py`); **not** frozen eval | Restore code-faithful equation. |
| Priority Buffer | IMPLEMENTED in UI; **not** frozen eval | Restore code-faithful rule. |
| Uncertainty \(U(r)\) | HISTORICAL paper-only | Do not restore as implemented. |
| Fail-closed gates | IMPLEMENTED (path-dependent) | OK. |
| Jetson | PLANNED | OK. |
| Metric metres | UNSUPPORTED | Forbidden. |

---

## 4. Terminology dictionary (canonical)

| Term | Meaning |
|------|---------|
| ARTPS | Autonomous Rover Target Prioritization System (architecture + software) |
| science-interest target | Human operational label under v1 annotation guide |
| anomaly / anomaly score | Cue of unusual appearance; not automatically a science target |
| candidate | Localized region after fusion |
| target / priority | Ranked candidate for review / downlink |
| curiosity score \(C(r)\) | UI ranking combination (known + anomaly + optional terms) |
| image score | Frozen eval: `max_valid_candidate_after_masks` |
| relative depth | Within-image near/far ordering (DPT); not metres |
| depth cue | Depth-edge / proximity weighting on maps |
| apparent size | Image extent + relative depth band; not physical size |
| entropy-weighted fusion | **Specified** Shannon-entropy cue weights (historical MS) |
| fixed-weight fusion | **Current detection-core** linear mix (frozen eval) |
| feature-space clustering / diversity penalty / soft similarity | UI ranking anti-redundancy |
| Priority Buffer | UI second-chance set after diversity penalty |
| operational mask / suppression | Rover-body, shadow, boundary, validity gates |
| historical reported evaluation | Accepted-abstract 0.894… (Layer A) |
| current reproducible validation | v1 protocol pin; test closed |
| supplementary evaluation | v1_1 human-reviewed validation audit (Layer C) |

Drift to fix: “independent evaluation” used for both v1 protocol and v1_1 supplementary table; “accuracy” vs ranking; “distance” vs relative depth (mostly already bounded).

---

## 5. Equation inventory

Current IAC TeX: **no display equations**.

32p / `8f7e3ff:docs/paper.tex` ARTPS-specific (restore only if code-faithful + labeled):

| Equation | 32p | Current code | Restore now? |
|----------|-----|--------------|--------------|
| Entropy \(H_i\), \(w_i\), \(A_{\mathrm{comb}}\) | yes | **absent** (fixed weights) | Yes, labeled **specified / not frozen-eval default** |
| Curiosity \(C(r)\) | yes | `curiosity_scorer.py` (UI) | Yes, labeled **ranking / UI; not frozen image_score** |
| Soft sim \(C'=C(1-\lambda s)\) | yes | `app.py` | Yes, same label |
| Buffer \(\tau_{\mathrm{high}},\tau_\Delta\) | yes | `app.py` | Yes, same label |
| \(U(r)\) disagreement | yes | **absent** | No |
| DHE \(g(p)\), \(I_{\mathrm{dhe}}\) | yes | optional enhancement path | Defer (P3); not frozen eval |
| Textbook P/R / ViT / FFN | 32p | n/a | Do not restore |

---

## 6. Results / Discussion / Limitations / figures / template / refs

**Results:** Historical vs v1_1 correctly separated; no apples-to-apples “0.772 replaces 0.894”; no F1 0.920 headline; test unopened. Table 2s still useful if score contract is explicit. Keep table.

**Discussion:** Prevalence bound to Mastcam sample is good. Missing operational *why* (hybrid vs anomaly-only; diversity; buffer). Entropy/code gap unstated.

**Limitations:** Checklist mostly complete. Add frozen-eval ranking-stack omission.

**Figures:** Fig 1 = tabular placeholder (`PRESENT` as schematic, not a drawing). Figs 2–5 `MISSING` (plan-only). Tables 1 + 2s `PRESENT`. Tables 2–6 `PLACEHOLDER`/plan-only. Do not invent figures this PR.

**Template:** Header/footer/ID/10pt Times/25 mm/letter — OK. Underfull hboxes many; no overfull on current 6p snapshot. `CORRESPONDING_EMAIL_TBD` allowed for planning milestone.

**Refs:** 4 cited keys, 0 unused. `estlin2012` incomplete. No fabricated DOI this PR.

**AI declaration:** consistent with author boundary. **STOP if editing meaning.** No edit.

---

## 7. Findings

### P0 — submission integrity

**P0-01**
- File: `sections/methods.tex` (also Intro contribution 2)
- Issue: Entropy-weighted fusion stated as what ARTPS uses.
- Evidence: `src/artps_detection_core.py` ~1043–1052 fixed `w_recon/w_depth/w_texture`; no Shannon entropy in `.py`. Ledger C02 claim text vs code.
- Why: Reviewers can treat entropy fusion as measured live behaviour.
- Fix: Specify entropy rule as **architectural specification**; state current frozen path uses **fixed-weight** multi-cue fusion.

**P0-02**
- File: `sections/results.tex`, `sections/methods.tex`
- Issue: v1_1 AUROC/AP come from frozen `artps_full_frozen_mars_clf_on_v1` image scores (`max_valid_candidate_after_masks`). Curiosity, diversity, and Priority Buffer are **not** in that path.
- Evidence: `src/artps_inference.py` vs `app.py` ranking; `INDEPENDENT_EVAL_V1_1_REPORT.md`.
- Why: Readers may attribute 0.772 to the full abstract stack.
- Fix: One explicit sentence in Results + Methods overview.

### P1 — should fix before submission

**P1-01** Intro contribution 2: qualify entropy vs current fusion.
**P1-02** Methods: name curiosity score \(C(r)\) to match abstract; UI-only / not frozen image_score.
**P1-03** Methods: restore code-faithful soft-similarity + Priority Buffer equations, same qualifier.
**P1-04** Results: state frozen score contract; keep AUROC 0.772 / AP 0.956 / no 0.920.
**P1-05** Discussion: one *why* block (hybrid, diversity, buffer); drop duplicate 340/20; mention entropy spec vs fixed-weight eval.
**P1-06** Experiments metrics: F1 is protocol-named; **not** a v1_1 headline metric.
**P1-07** Ledger C02 **notes** (support class unchanged unless tests allow): entropy specified; current core fixed-weight.
**P1-08** Results subsection title / prose: “independent evaluation” vs “supplementary validation”.
**P1-09** Limitations: frozen eval does not exercise curiosity/diversity/buffer.

### P2 — polish (this PR only if low-risk)

Repeated safety paragraphs; Fig 1 placeholder; `estlin2012` author expansion (unverified locally in this environment → defer); Related Work TODO cites; abstract/title strong words (accepted — do not change); underfull boxes; Discussion “useful ranking”; ledger taxonomy “dataset not yet present”.

### P3 — optional later

Email placeholder; real architecture figure; Tables 2–6; Jetson; Estlin 2014; DHE / \(U(r)\) equations; camera-ready DOIs; independent second annotator; historical reproduction closure.

---

## 8. Minimal patch plan (this PR)

1. Write this audit + `AUDIT_ACCEPTED_ABSTRACT_ALIGNMENT.md`.
2. Patch Methods / Intro / Results / Discussion / Limitations / Experiments / C02 notes only.
3. Restore only entropy (labeled specified), curiosity, similarity, buffer equations.
4. Do **not** change abstract text/numbers, Table 1, 0.772/0.956, declaration, freeze, F1 artifacts.
5. Extend manuscript tests for new invariants.
6. Build + CI + merge.

Invariants unchanged: historical YES/NO report fields must remain **NO** for metric edits.

---

## 9. Post-patch status (same PR)

Implemented: P0-01, P0-02, P1-01–P1-09, plus Fig.~1 schematic label “Fusion” (not “Entropy fusion”).
Deferred: all P2/P3 except Related Work “peak accuracy” → “ranking discrimination”.
Abstract / declaration / freeze / historical numbers / v1_1 0.772+0.956 unchanged.

---

## 10. Camera-ready status (PR after #32)

Branch: `paper/iac2026-camera-ready-assets`. Does not reopen P0/P1 evidence issues.

REAL_ARCHITECTURE_FIGURE: complete
(TikZ `figures/artps_pipeline.tex`; layers A/B/C; fixed-weight vs dashed entropy spec; ranking not in frozen v1_1 image score)

BIBLIOGRAPHY: verified for 4 cited keys (`estlin2012` → AEGIS ACM TIST 2012; PaDiM; PatchCore; DPT).
Unresolved: Estlin ISAIRAS 2014 = UNVERIFIED_METADATA (no DOI/pages; not added).

TODO_CITATIONS: PDF-visible = 0; comment-only = 7
(1× methods AE/FastFlow/CFA planning cite; 4× `% TODO(results)`; 1× author email; 1× declaration slot)

EMAIL: complete (`poyrazbaydemir@gmail.com`; author-supplied)

P0: 0

P1: 0

P2: remaining 3
(underfull hboxes harmless; accepted title/abstract strong words leave-as-is; ledger taxonomy note out of scope)

CAMERA_READY: ready_except_author_email

Tables 2–6 / Figs 2–5: DEFER (no invented values). Table 1 + Table 2s KEEP.

Legacy extended-MS assets: see [`LEGACY_EXTENDED_MANUSCRIPT_ASSET_AUDIT.md`](LEGACY_EXTENDED_MANUSCRIPT_ASSET_AUDIT.md). Recommended decision **B** (architecture figure only) pending author review. No TeX/figure change in that audit PR.
