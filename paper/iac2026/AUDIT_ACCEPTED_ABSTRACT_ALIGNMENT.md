# Accepted-abstract ↔ full-paper alignment

Source abstract: `paper/iac2026/main.tex` `\IACmaketitle` body (same numbers as Portal/T3 abstract PDFs under `docs/`).
Historical 32p: `docs/Full_Baydemir_ARTPS.pdf` (same title + Table 2 metrics).
Full paper body: `paper/iac2026/sections/*.tex` after PR #31.

Do **not** expand ARTPS into a different system. Abstract numbers stay 0.894 / 0.847 / 0.823 / 0.856 / 28.1.

| # | Abstract sentence (condensed) | Body locus | Class | Note |
|---|-------------------------------|------------|-------|------|
| 1 | Bandwidth/energy + delayed ground → onboard prioritization of scientific targets | Intro ¶1; Methods overview | ALIGNED | |
| 2 | Visual-only autonomy brittle: illumination, shadows, low texture → FPs; diversity can suppress similar high-value anomalies | Intro ¶2; Methods suppression + diversity + buffer | ALIGNED | Diversity/buffer *why* is thin in Discussion (P1-05). |
| 3 | ARTPS couples monocular **relative-depth** with multi-cue anomaly detection and localization | Intro; Methods depth + anomaly; Related Work DPT | ALIGNED | Body correctly forbids metric distance. Abstract “robust localization” is qualifier-only. |
| 4 | Entropy-weighted dynamic fusion of image and depth cues | Intro contrib 2; Methods fusion | **POTENTIAL DRIFT** | Abstract = accepted Layer A specification. **Body currently reads as live implementation.** Code = fixed-weight fusion. Patch body, not abstract. |
| 5 | Ranks with **curiosity score** and **soft similarity penalty** | Abstract only vs Methods diversity prose | **POTENTIAL DRIFT** | Mechanisms exist in UI (`curiosity_scorer.py`, `app.py`). Methods never names curiosity. Frozen eval **does not** apply them. |
| 6 | Priority Buffer for second-chance re-eval of high-anomaly targets down-weighted by diversity | Methods Priority Buffer | ELABORATION | Prose present; no equation; not in frozen eval. |
| 7 | Operational gates: rover-body, boundary-shadow, objects-in-shadow, image-relative size–distance (not metric) | Methods masks + size–distance; Experiments proxies | ALIGNED / ELABORATION | Proxy evidence C09–C11 stays non-headline. |
| 8 | NASA Mars rover imagery: 0.894 AUROC, 0.847 AUPRC, 0.823 F1 vs PaDiM/PatchCore 0.856 | Table 1; Experiments historical track; ledger C05–C06 | ALIGNED | Framed as accepted-abstract, reproduction pending. **Not** replaced by 0.772. |
| 9 | Lightweight core (no learned depth/AE) 28.1 FPS @ 256×256 workstation | Table 1 FPS row; Experiments hardware; C07 | ALIGNED | Body does not claim full neural pipeline = 28.1. |
| 10 | Keywords: Mars rover; autonomous exploration; anomaly detection; relative depth; target prioritization; operational safety | Keywords unchanged | ALIGNED | “Autonomous” = screening, not replacing mission planning (Related Work). |

## Required theme checklist

| Theme | Present in body? | Status |
|-------|------------------|--------|
| ARTPS motivation (bandwidth / delayed loop / FPs) | Yes | ALIGNED |
| Monocular relative depth + multi-cue anomaly | Yes | ALIGNED |
| Entropy-weighted dynamic fusion | Named, **implementation mismatch** | POTENTIAL DRIFT → patch Methods/Intro |
| Feature-space clustering | Named lightly (“feature space”) | ELABORATION |
| Soft similarity penalty | Implied as “penalizes … similar”; no formula | POTENTIAL DRIFT → restore equation + UI qualifier |
| Priority Buffer | Yes | ELABORATION → equation + not-in-frozen-eval |
| Historical 0.894 / 0.847 / 0.823 / 0.856 | Table 1 + abstract | ALIGNED |
| 28.1 lightweight qualification | Table 1 + abstract | ALIGNED |

## Supplementary (not in accepted abstract)

`independent_eval_v1` / `v1_1` validation audit (360 review; 340/20; AUROC 0.772; AP 0.956) is **SUPPLEMENTARY**. Must remain labeled as such. Must not migrate into abstract.

## Drift rule for this PR

- **Do not** change abstract wording or numbers (accepted Layer A).
- **Do** make the body honest about: entropy = specified; frozen eval = fixed-weight + `max_valid_candidate_after_masks`; curiosity/similarity/buffer = operational ranking (UI), not v1_1 table.
