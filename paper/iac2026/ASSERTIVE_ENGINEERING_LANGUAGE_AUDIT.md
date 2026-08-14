# Assertive Engineering Language Audit

**Date:** 2026-08-14  
**Base commit:** `93a8da6` (post PR #58 / `main` — architecture + citation polish)  
**Scope:** Reviewer-visible TeX only (`main.tex` abstract + `sections/*.tex`). TeX not rewritten in this artifact.

**Pre-rewrite inventory (non-comment lines):** ~87 standalone `\bnot\b` hits across abstract + sections; hotspots `methods.tex` (~40) ≫ `experiments.tex` / `discussion.tex` / `limitations.tex`. Approximate class mix before rewrite: **~55–65 A**, **~15–20 B**, **~10–15 C** (retain/rephrase), **~5–10 D** (math / comments / irrelevant). Goal after rewrite + theme dedupe: reviewer-visible defensive negation as close to zero as technically possible while freezing science.

**Problem types:**  
- **A** = DEFENSIVE NEGATION — rewrite to scope-first prose  
- **B** = NECESSARY TECHNICAL CONTRAST — prefer rewrite when affirmative scope is clearer  
- **C** = PRECISE TECHNICAL TERM — retain meaning; rephrase affirmatively where listed (do not delete)  
- **D** = MATHEMATICAL / bib / irrelevant — ignore  

**Scientific freeze (all replacements must preserve):** relative depth non-metric; Layer C downstream of image-level score; reserved 54-test; candidate = localization hypothesis; no flight certification; qualitative ≠ quantitative; metrics/equations/figure topology unchanged.

**Abstract policy:** Only the parenthetical `(not metric distance)` is A. Leave PR #58 architecture sentences alone.

---

## Abstract (`main.tex`)

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Abstract | Operational gates … apply an image-relative size--distance policy based on apparent size **(not metric distance)**. | A | Relative depth / size policy non-metric; metrics unchanged | … based on **apparent size under image-relative depth**. | Scope-first quantity definition; sole abstract touch per PR #58 policy. |
| Abstract | ARTPS combines … Priority Buffer … (architecture block) | D | PR #58 approved wording | *(leave unchanged)* | Architecture sentences frozen; not part of this negation pass. |
| Abstract comments / title block | `% Do not rewrite Results numbers…` etc. | D | Editorial comments | *(ignore)* | Non-reviewer-visible. |

---

## Introduction (`sections/introduction.tex`)

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Intro | **Not** every acquired image carries equal scientific value … prioritize candidate targets **rather than** treating all detections as equally actionable. | B | Prioritization under bandwidth | Acquired images differ in follow-up value, so onboard systems **prioritize candidate targets by evidence strength**. | Affirmative operational need; drop opening negation. |
| Intro | Classical visual anomaly detection … alone it **does not** define an operational target-selection policy. | A | Anomaly ≠ full policy | Classical visual anomaly detection supplies appearance cues; **operational target selection additionally requires** localization, suppression, diversity, and ranking. | Define missing stack instead of “does not define”. |
| Intro | Monocular depth estimators supply image-relative near/far structure **rather than** calibrated metric distance … | B | Relative, non-metric depth | Monocular depth estimators provide **image-relative near/far structure for ordering and policy decisions**. | Preferred Intro depth wording; cite remains. |
| Intro | The architecture … **is not** flight-qualified hardware and **does not** assert universal safety guarantees. | A | No flight certification | The present study evaluates ARTPS as a **safety-aware, flight-readiness-oriented software architecture within a defined operational envelope**. Flight qualification and mission certification require **independent mission-specific V&V**. | Certification envelope once, affirmative; no stronger claim. |

---

## Related Work (`sections/related_work.tex`)

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| RW 2.1 | anomaly cues feed prioritization **rather than** replacing mission planning. | B | Human/mission planning remains | Anomaly cues **feed prioritization inside** the broader mission-planning loop. | Affirmative framing of role. |
| RW 2.2 | a high anomaly score **is not identical to** an operationally actionable science target under rover constraints | A | Candidate / cue ≠ confirmed target | **Operational actionability additionally depends on** localization, false-positive suppression, diversity, and ranking under rover constraints. | Preferred RW meaning. |
| RW 2.3 | They **do not**, by themselves, provide metric distance, physical object size, or calibrated 3D reconstruction… | A | Non-metric depth | ARTPS interprets monocular depth as **image-relative scene structure**; metric range, physical dimensions, and calibrated 3-D geometry **require calibrated sensing**. | Scope-first sensing boundary. |
| RW 2.3 | ARTPS uses relative depth as an ordering and policy cue; **no metric-distance claim is made**. | A | Non-metric depth | ARTPS therefore uses depth **exclusively as an image-relative ordering and policy cue**. | Define use; drop “no claim”. |
| RW 2.4 | … bounded assurance … **not** universal or flight-certified guarantees. | A | No flight certification | Assurance is **bounded to the defined software operational envelope**; flight certification requires mission-specific V&V. | Preferred RW certification wording. |
| RW 2.4 | fail-safe ``**no selection**'' behaviour | C | Fail-closed output token | Keep **`no-selection` / no selection** as the operational state name. | Precise technical term. |

---

## Methods — Fig. 1 caption & overview (`sections/methods.tex`)

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Fig. 1 cap. | The optional classifier contributes a known-value scalar to candidate-level scoring, **not to** fixed map fusion. | A / C | Classifier off map fusion | The classifier contributes a known-value scalar **directly to candidate scoring**. | Positive topology; Layer B/C C-boundary companion. |
| Fig. 1 cap. | Layer~C … **is not included** in the image-level score. | A / C | Layer C downstream | Layer~B **terminates at** the image-level score, while Layer~C **consumes valid candidates** for curiosity-, diversity-, and buffer-based operational ranking. | Preferred Fig. 1 affirmative boundary. |
| Overview | … or an explicit **no-selection** outcome when evidence is insufficient. | C | Fail-closed | Retain **`no-selection`**. | Operational output token. |
| Overview | Curiosity ranking, feature-space diversity, and the Priority Buffer … are \emph{**not**} applied on the image-level score. | A / C | Layer C downstream | **Image-level scoring terminates at Layer~B.** Curiosity, feature-space diversity, and the Priority Buffer **operate subsequently in Layer~C**. | Preferred architectural boundary. |
| Overview | Optional Mars-oriented enhancement … is preprocessing, **not a** detection metric. | A | Enhancement ≠ metric | Optional Mars-oriented enhancement is **confined to preprocessing upstream of cue extraction**. | Preferred overview wording. |
| Overview | Optional super-resolution enhancement **was disabled** in all reported experiments. | B | Config fact | Reported experiments use the **native-resolution preprocessing configuration, with the super-resolution module inactive**. | Configuration-first (acceptable B→assertive). |

---

## Methods — Multi-cue & depth

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Multi-cue | Fixed-coefficient multi-cue fusion uses a relative-depth \emph{edge} cue …, **not a** protrusion map. | B / C | Depth-edge ≠ protrusion | Fusion uses a relative-depth **edge** (image-relative discontinuity) cue; **protrusion remains a separate geomorph descriptor outside the fused score**. | Keep contrast as cue definition; affirmative. |
| Multi-cue | The depth-edge cue highlights regions … **rather than** from appearance alone. | B | Cue complementarity | The depth-edge cue highlights structure **relative to the surrounding near/far layout**, complementary to appearance cues. | Affirmative complementarity. |
| Multi-cue | **No** single cue is treated as a mineral detector, lithology classifier, or complete science label… | A | Cue role only | **Each cue serves as an input to fusion and localization.** Mineralogical or lithological interpretation requires dedicated sensing and labels beyond these anomaly cues. | Preferred cue-role wording. |
| Multi-cue | Fig. qualitative(b) … on a **non-test** example. | C | Qualitative ≠ quantitative | Retain **`non-test`**; pair with caption quantitative→Tables pointer (theme C). | Precise technical term. |
| Depth stack (4 sentences) | … **rather than** absolute metric range. / … **not as** metric metres… / … **not** transferable … / **no** metric-depth claim is made. | A | Relative depth non-metric | **Consolidate to one definition:** ARTPS maps monocular depth to a **within-image relative field** used for near/far ordering, depth-edge extraction, and apparent-size policy. The representation is **non-metric**; absolute range and physical dimensions require calibrated geometry. | Preferred single definition; kill disclaimer stack. |
| Depth | Keep word **non-metric** where retained after consolidate | C | Non-metric | Retain **`non-metric`**. | Precise technical term. |

---

## Methods — Fig. 2 / Fig. 3 captions & fusion

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Fig. 2 cap. | … it **is not a** pixel-level segmentation or ground-truth object boundary. | A | Overlay = localization support | The translucent footprint visualizes **candidate proposal-support geometry at image-region level**. | Preferred Fig. 2 semantics. |
| Fig. 2 cap. | The depth visualization is relative and **non-metric**. | C | Non-metric | Retain; optionally fold into Methods depth definition to cut repetition (theme A). | Technical term. |
| Fig. 2 cap. | The example is illustrative and **is not used as** an additional quantitative benchmark. | A | Qualitative ≠ quantitative | The example provides **qualitative process visualization**; quantitative evaluation is reported separately in Tables~\ref{tab:accepted-abstract}--\ref{tab:indep-v11}. | Preferred figure semantics. |
| Fig. qualitative(c) | shows **only** this image-relative near/far field. | B | Panel scope | Panel (c) displays the **image-relative near/far field**. | Drop restrictive “only” if redundant after caption. |
| Fusion | … applies a \emph{fixed-coefficient} linear mix **instead of** this entropy rule. | B | Fixed-coeff measured path | The human-reviewed validation applies a **fixed-coefficient** linear mix; entropy-weighted fusion remains the adaptive specification. | Configuration scope without negation frame. |
| Fusion eq. | \(P_{\mathrm{mix}}=0.55\,P+0.45\,(1-P)\) | D | Math | *(ignore)* | Mathematical \((1-P)\). |
| Fusion | … they **are not required** to sum to one… | A | Coefficient semantics | The coefficients are **implementation weights** followed by proximity modulation and clipping; **unit-sum normalization is therefore unnecessary**. | Preferred fusion wording. |
| Fusion | They **are not claimed** to be optimal or learned. | A | Fixed manual coeffs | The evaluated configuration uses **fixed, manually specified coefficients**. | State what they are. |
| Fusion | … far-field responses are down-weighted **rather than** zeroed. | B | Proximity band | Far-field responses remain in a **narrow proximity band near 0.75** (down-weighted, nonzero). | Affirmative policy. |
| Fusion | Layer~C ranking … **is not applied** when forming the image-level score. | A / C | Layer C downstream | **Layer~B produces the image-level score; Layer~C operates on the surviving candidate set** for downstream ranking. | Preferred Layer boundary. |
| Fig. 3 cap. | Smaller … terms … **are not shown** as separate panels. | A | Display selection | The four displayed panels emphasize the dominant reconstruction, relative-depth-edge, texture, and fused responses; depth-Laplacian and fine-detail terms **also contribute to the fusion**. | Preferred Fig. 3 display wording. |
| Fig. 3 cap. | … color intensities are therefore **not numerically comparable** across panels. | A | Viz scale | Independent min-max display scaling assigns each panel its own visualization range; cross-panel color intensity **serves qualitative inspection**. | Preferred comparability wording. |
| Fig. 3 cap. | … **non-metric**. / … **is not used as** an additional quantitative benchmark. | C / A | Non-metric; qual ≠ quant | Retain **non-metric**; replace benchmark negation with Tables pointer (as Fig. 2). | C + A. |
| Cue text | … discontinuity cue, **not** metric distance and **not** protrusion. | B / C | Edge ≠ metric/protrusion | Panel (b) is an **image-relative discontinuity cue** for fusion (protrusion remains outside the fused score). | Affirmative cue ID. |
| Cue text | … **it is not a** term in the fused map. | A | Classifier off fusion map | The classifier / known-value term enters **candidate scoring and curiosity**; the fused map uses reconstruction, depth-edge, texture, and related cues only. | Positive topology. |
| Cue text | Display scaling … **does not** constitute a new quantitative experiment. | A | Qual ≠ quant | Display scaling is **independent visualization**; quantitative evaluation remains Tables~\ref{tab:accepted-abstract}--\ref{tab:indep-v11}. | Scope-first. |

---

## Methods — Localization, suppression, scoring, ranking

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Localization | Candidates that fail validity checks … **do not** contribute to the image-level score. | A / C | Max-over-valid | **Validity gates restrict image-level aggregation to surviving candidates.** | Preferred localization wording. |
| Localization | … maximum among remaining valid candidates, **not a** mean … and **not a** curiosity-weighted sum. | A / C | Aggregation = max | Image-level aggregation uses the **maximum candidate score over \(R_{\mathrm{valid}}\)**; curiosity-weighted ranking occurs later in Layer~C. | Preferred aggregation definition. |
| Localization | Fig. qualitative(d) … **is not a** downlink ranking and **does not** apply curiosity, diversity, or the Priority Buffer. | A / C | Overlay = Layer B | Fig.~\ref{fig:qualitative}(d) visualizes the **Layer~B surviving-candidate set prior to Layer~C ranking**. | Preferred panel semantics. |
| Suppression | … responses that **do not** correspond to science-interest targets. | B | Distractor inflation | Distractor regions can produce **high reconstruction or texture responses unrelated to science-interest targets**. | Mild B rewrite. |
| Suppression | … including cast-shadow wedges … **rather than** suppressing all dark pixels. | B | Boundary-shadow policy | Boundary-connected shadow suppression targets **dark, low-structure, boundary-connected** artefacts (e.g. cast-shadow wedges). | Affirmative target set. |
| Suppression | … suppressing the artefact **must not** automatically discard the in-shadow object. | A | Object-in-shadow gate | The object-in-shadow gate **preserves structured candidates within shadowed regions** while suppressing flat boundary-shadow artefacts. | Preferred gate wording. |
| Suppression | They reduce invalid candidates; they **do not** establish that a surviving region is scientifically relevant. | A | Candidate = hypothesis | The masks enforce **candidate-validity constraints**; science-interest confirmation occurs during subsequent operational interpretation. | Preferred suppression scope. |
| Suppression | … overlay … **not a** curiosity-ranked downlink list. | A | Layer B overlay | … shows the **Layer~B candidate-support overlay** on one non-test example. | Affirmative. |
| Suppression | … localization hypothesis: it **is not a** confirmed science target and **is not** automatically queued for downlink. | A | Candidate = hypothesis | A surviving candidate remains a **localization hypothesis** and proceeds to **downstream ranking before downlink selection**. | Preferred output semantics. |
| Suppression | … **it is not a** claim of perfect distractor rejection. | A | Residual risk | Suppression is **policy-driven and inspectable**, with residual distractor risk handled by downstream ranking and mission-level V&V. | Preferred residual-risk wording. |
| Score \(K\) | … **not a** per-box classifier and **not a** claimed lithology label. | A | \(K\) semantics | \(K\) is an **image-global known-value scalar** used exclusively as a candidate-scoring feature; its semantics **exclude per-box and lithological interpretation**. | Preferred \(K\) wording. |
| Score eq. | \(0.05\,(1-D_r)\) | D | Math | *(ignore)* | Mathematical \((1-D_r)\). |
| Score | These terms **were disabled** … therefore contribute zero… | A | Fixed-to-zero | The human-reviewed Layer~B configuration **fixes the PaDiM and PatchCore score terms to zero**; PaDiM/PatchCore performance is reported through the **separate baseline evaluation**. | Preferred config language. |
| Score | … **no** additional score formula is introduced. | A | Sole score eqs | Recall-support pooling modifies \(F_r\) upstream while the equations above remain the **sole candidate-score formulation**. | Preferred. |
| Size–distance | This is a relative heuristic, **not a** physical size measurement in metres. | A | Non-metric size | The policy operates in **normalized image/depth coordinates**; metric physical size requires calibrated geometry. | Preferred size policy. |
| Ranking | This operational ranking path **is not part of** the image-level score. | A / C | Layer C downstream | **Image-level scoring ends before** the Layer~C operational-ranking sequence. | Preferred. |
| Curiosity | \(C(r)\) is \emph{**not**} the image-level score. | A / C | Layer C | \(C(r)\) is **reserved for Layer~C operational ranking**. | Preferred. |
| Diversity | \(S'_r=S_r\,(1-\lambda s(r))\) | D | Math | *(ignore)* | Mathematical \((1-\lambda s)\). |
| Diversity | This ranking step **is not applied** when computing the image-level score. | A / C | Layer C | **Diversity acts after Layer~B image scoring.** | Preferred. |
| Buffer | … the buffer **does not** recover mask-invalidated regions. / … **not a** guarantee … / **not part of** the image-level score. | A / C | Buffer ⊆ Layer C; valid-only | Priority Buffer eligibility is restricted to candidates that **pass Layer~B validity gates**. The buffer provides **second-chance ranking within Layer~C**, while the image-level score remains fixed at the Layer~B output. | Preferred consolidated buffer wording. |
| Fail-closed | returns an explicit **no-selection** outcome… | C | Fail-closed | Retain **`no-selection`**. | Technical term. |
| Fail-closed | … **rather than** emitting an unverified priority list. | A | Abort semantics | Incomplete required inputs trigger **deterministic abort / no-selection** behaviour. | Preferred runtime wording. |

---

## Experimental Protocol (`sections/experiments.tex`)

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Protocol | … illustrates the image-level scoring path only; it **is not a** third quantitative evaluation. | A | Qual ≠ quant | The quantitative study comprises the **two components** in Table~\ref{tab:eval-tracks}; Fig.~\ref{fig:qualitative} provides **qualitative process visualization**. | Preferred protocol wording. |
| Human review | This **is not** an independent second annotation. | A | Repeat-author review | The annotation protocol is a **model-blind repeat-author review**. | Preferred. |
| Human review | The 54-image test partition was reserved and **was not used** for ARTPS inference, threshold selection, or … reporting. | A / C | Reserved 54-test | Human-reviewed performance is reported on the **54-image validation partition**; the corresponding **54-image test partition remains reserved**. | Preferred reserved-test pattern. |
| Human review | … with **no** additional tuning and **without** repeating inference. | B | Frozen inference | Numeric validation results use the reviewed labels under the **frozen inference and score definition**. | Affirmative freeze. |
| Fig. close-far | … **it is not a** range-conditioned performance study. | A | Qual scale context | Fig.~\ref{fig:close-far} provides **qualitative apparent-scale context** across close and distant scenes; range-conditioned performance remains outside this analysis. | Preferred. |
| Dataset | … this **is not** independent double annotation. | A | Repeat-author | The labels represent **repeat-author model-blind review**. | Preferred. |
| Dataset | A negative label denotes **no** distinct science-interest target… | C / B | Label semantics | Retain logical negative definition **or** rephrase as inclusion: positives require a distinct science-interest target; negatives are images **lacking** such a target under the guide. | Keep meaning; soften if easy. |
| Dataset | Rover hardware … alone **do not** count as positive. | B | Inclusion criteria | Positive labels require a **real surface object/region** meeting the guide; rover hardware, border/telemetry artefacts, compression/noise, shadow boundary alone, or exposure/specular artefact alone **fall outside** the positive class. | Inclusion-first. |
| Table 1 | reserved test **unused** | A / C | Reserved 54-test | **54-image validation analysis; 54-image test reserved** | Preferred table cell. |
| Table 1 | Layer~C ranking **not applied** to these metrics | A / C | Layer C downstream | **Layer~B detection metrics; Layer~C downstream** | Preferred table cell. |
| Table 1 | Curiosity, diversity, and Priority Buffer **excluded** from this score analysis | A / C | Layer C downstream | **Operational ranking downstream of reported image-score analysis** | Preferred table cell. |
| Baselines | PaDiM and PatchCore … **are not** active components of the human-reviewed image-level score. | A | Fixed-to-zero | The human-reviewed Layer~B configuration **sets the PaDiM/PatchCore score terms to zero**; baseline performance is evaluated separately. | Preferred. |
| Metrics | threshold-dependent F1 **is not interpreted** as a performance measure… | A | All-positive OP | The all-positive operating point renders threshold-dependent F1 **uninformative** for this validation; AUROC and AP therefore characterize ranking performance. | Preferred. |
| Repro | reserved 54-image test … **was not used** for … | A / C | Reserved 54-test | Same reserved-test affirmative pattern as above (dedupe theme D — keep one instance in Protocol + Limitations). | Theme D. |
| Hardware | Full-pipeline edge-hardware timing … **was not evaluated** … **No** flight qualification is claimed. | A | No flight cert; Jetson future | The 28.1\,FPS measurement characterizes the **lightweight preprocessing–fusion–localization core** on the development workstation. Full-pipeline edge-hardware timing and flight qualification constitute **separate hardware and mission V&V stages**. | Preferred hardware/cert scope. |

---

## Results (`sections/results.tex`)

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Human-reviewed | Curiosity ranking, diversity penalty, and the Priority Buffer **were not applied** on these scores. | A / C | Layer C downstream | These results characterize the **Layer~B image-level score before Layer~C operational ranking**. | Preferred Results boundary. |
| Human-reviewed | … threshold-dependent classification metrics **are not treated as** performance measures… | A | All-positive OP | The all-positive operating point makes threshold-dependent classification metrics **non-discriminative**; AUROC serves as the principal ranking measure. | Preferred (retain “non-discriminative” as technical). |
| Qualitative | Fig. cue-decomp … **it is not a** new detection metric. | A | Diagnostic decomp | Fig.~\ref{fig:cue-decomp} is a **diagnostic decomposition of the existing Layer~B fusion**. | Preferred. |
| Fig. close-far cap. | … **is not a** pixel-level segmentation or ground-truth object boundary. | A | Overlay semantics | … the translucent footprint shows **supporting image-region evidence** (proposal-support geometry). | Align with Fig. 2 preferred. |
| Fig. close-far cap. | … **are not presented as** an additional quantitative benchmark or a controlled range-conditioned performance study. | A | Qual ≠ quant | The examples provide **qualitative apparent-scale context**; quantitative evaluation is reported in Tables~\ref{tab:accepted-abstract}--\ref{tab:indep-v11}. | Scope-first. |
| Results body | … **non-test** illustration / **non-test** scenes | C | Qual scenes | Retain **`non-test`**. | Technical term. |

---

## Discussion (`sections/discussion.tex`)

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Discussion | … with explicit **no-selection** when evidence is insufficient. | C | Fail-closed | Retain **`no-selection`**. | Technical term. |
| Discussion | The design goal is structured prioritization under rover constraints, **not** unconstrained claim of superior detection in all regimes. | A | Bounded envelope | The design targets **structured prioritization within a bounded rover operating envelope**. | Preferred. |
| Discussion | … a high cue **is not** automatically a science-interest target… | A | Candidate = hypothesis | **Operational target selection combines** anomaly response with validity, spatial context, suppression, and ranking. | Preferred. |
| Discussion | Relative depth supplies near/far structure … **without** metric distance… | B | Non-metric | Relative depth supplies **image-relative near/far structure** for apparent-size policy. | Affirmative; cite stays. |
| Discussion | These ranking mechanisms … **are not part of** the image-level validation scores. | A / C | Layer C downstream | The reported image-level validation **terminates at Layer~B**; Layer~C supplies downstream operational ranking. | Preferred (dedupe theme B). |
| Discussion | Fig. qualitative … Layer~B products … **rather than** a Priority-Buffer or diversity-ranked target list. | A | Layer B viz | Fig.~\ref{fig:qualitative} visualizes **Layer~B products**: the post-suppression combined map, relative depth, and surviving candidates. | Preferred. |
| Discussion | Fig. cue-decomp … **it does not** introduce a new metric. | A | Diagnostic | Fig.~\ref{fig:cue-decomp} **decomposes the existing Layer~B fusion** into its principal visual cues. | Preferred. |
| Discussion | … a high local cue or a near-field blob **does not** by itself become a science target, and a single **non-test** frame **cannot** stand in for a detection benchmark. | A | Hypothesis + qual | Candidate actionability emerges from the **full validity and ranking sequence**. The single-scene example serves **qualitative mechanism inspection**; benchmark evidence is provided by the quantitative evaluation. | Preferred split. |
| Discussion | … **not a** pixel-level segmentation, **not a** measured curiosity rank, and **not a** claim that the region is scientifically confirmed. | A | Localization hypothesis | The overlay represents a **Layer~B localization hypothesis** supported by candidate-region evidence. Curiosity ranking and science-target confirmation occur **downstream**. | Preferred triple-negation kill. |
| Discussion | … it **should not** be generalized to planetary imagery as a whole. | A | Prevalence scope | This observed prevalence is **specific to** the sampled Curiosity Mastcam domain and the adopted annotation definition. | Scope-first prevalence. |
| Discussion | Threshold-based F1 **is not informative** here… | A | All-positive OP | Same uninformative-F1 / AUROC-AP ranking wording as Experiments. | Align sections. |
| Discussion | … \(28.1\)\,FPS … **is not a** full neural-pipeline edge-hardware or flight-qualification result. | A | Hardware / cert scope | The reported \(28.1\)\,FPS is a **lightweight-core workstation measurement**; full-pipeline edge timing and flight qualification remain **separate V&V stages**. | Align with Hardware preferred; dedupe theme E. |

---

## Limitations (`sections/limitations.tex`)

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Limitations | Monocular depth … is \emph{relative}, **not** metric: near/far ordering **does not** yield metres… | A | Relative depth non-metric | The ARTPS depth representation provides **relative near/far ordering**. Metric range, physical dimensions, and calibrated 3-D reconstruction **require calibrated geometry**. | Preferred; one Methods+Limitations reminder (theme A). |
| Limitations | The qualitative figures are **non-test** illustrations and **do not** represent field diversity or a controlled near/far study. | A / C | Qual ≠ quant | Figures~\ref{fig:qualitative}--\ref{fig:close-far} provide **qualitative mechanism illustrations**; their evidentiary role is **separate from** the quantitative evaluation. | Preferred safer wording; keep `non-test` if still needed once. |
| Limitations | … reserved 54-image test partition **was not used** for ARTPS inference or performance reporting. | A / C | Reserved 54-test | Human-reviewed performance is based on the **54-image validation partition**; the **54-image test partition remains reserved**. | Preferred pattern (theme D). |
| Limitations | … repeat-author review, **not** an independent second annotation. | A | Annotation protocol | The labels represent **repeat-author review**. | Preferred. |
| Limitations | The human-reviewed set contains **only** \(20\) negatives… / … contains **only** eight negative examples… | B | Prevalence / AUROC precision | State counts affirmatively: **20 negatives / 360**; validation has **eight negatives**, which **limits operating-threshold stability and AUROC precision**. | Facts stay; drop apology tone. |
| Limitations | The image-level score **does not** include curiosity ranking, diversity penalty, or the Priority Buffer. | A / C | Layer C downstream | The image-level score **terminates at Layer~B**; curiosity, diversity, and Priority Buffer **operate in Layer~C**. | Preferred. |
| Limitations | training-set lineage … **was not** independently cross-checked … **rather than** an external benchmark. | A | Internal validation | Checkpoint training-set lineage remains **unverified against** the human-reviewed corpus; the resulting evidence is categorized as **internal validation** rather than an external benchmark. *(“rather than” OK once as taxonomy)* | Preferred lineage wording. |
| Limitations | Full-pipeline edge-hardware timing … **was not evaluated**… | A | Jetson future | The current hardware evidence covers **lightweight workstation timing**; full-pipeline Jetson timing remains a **future hardware-validation stage**. | Preferred. |
| Limitations | It **is not** flight-qualified avionics, **not** mission-certified, and **does not** assert universal safety guarantees. | A | No flight certification | Current evidence covers a **safety-aware, flight-readiness-oriented software architecture within a defined operational envelope**. Flight qualification, mission certification, and system-level safety assurance require **independent mission-specific V&V**. | Preferred single certification sentence (theme E). |
| Limitations | Operational envelopes, independent V\&V, and mission-specific safety evidence remain **out of scope**. | B | Envelope | Fold into the certification sentence above if redundant. | Dedupe theme E. |

---

## Conclusion (`sections/conclusion.tex`)

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Conclusion | The human-reviewed validation uses the fixed-coefficient Layer~B image-score path **rather than** entropy-weighted fusion, and yielded AUROC~$0.772$… | B | Fixed-coeff measured; entropy = spec | The human-reviewed validation uses the **fixed-coefficient Layer~B configuration** and yielded AUROC~$0.772$; **entropy-weighted fusion remains the adaptive architectural specification**. | Preferred; avoid negation-framed contrast. |

---

## Declaration (`sections/declaration.tex`)

| Section | Original sentence | Problem type | Scientific invariant | Replacement | Reason |
| --- | --- | --- | --- | --- | --- |
| Declaration body | Generative AI tools were used solely for language verification… | D | Frozen AI disclosure | *(ignore / do not edit casually)* | Frozen disclosure prose. |
| Declaration comments | `% Language-verification disclosure only. Do not imply AI-authored science.` | D | Comment | *(ignore)* | Non-reviewer-visible; skip as D per task. |

---

## C-class retain checklist (affirmative rephrase, meaning kept)

| Token / boundary | Action |
| --- | --- |
| `non-metric` | Retain as technical descriptor of depth / size policy. |
| `non-test` | Retain for qualitative scenes; reduce repetition (theme C). |
| `no-selection` | Retain as fail-closed operational output. |
| Layer B terminates / Layer C downstream | Affirmative boundary wording; do not delete. |
| Reserved 54-image test | Affirmative “remains reserved”; do not delete. |
| Max-over-valid aggregation | State \(\max_{R_{\mathrm{valid}}}\); do not delete. |
| Depth-edge ≠ protrusion | Keep as cue definition in affirmative form. |
| Negative label counts / class definitions | Keep facts; prefer inclusion criteria. |
| Fixed-to-zero PaDiM/PatchCore in human-reviewed Layer B | State configuration affirmatively. |

---

## D-class ignore (non-exhaustive)

- Comments in `main.tex`, `declaration.tex`, figure TeX route files.
- Equations containing \((1-P)\), \((1-D_r)\), \((1-\lambda s(r))\).
- Bibliography / `\citep{...}` wrappers.
- PR #58 abstract architecture sentences (except metric-distance parenthetical).
- Figure TikZ comments (`artps_pipeline.tex` layout notes) — not body prose.

---

## Theme deduplication targets

After sentence rewrites, enforce preferred frequency (Phase 21):

| Theme | Content | Preferred frequency |
| --- | --- | --- |
| **A** | Relative depth = non-metric | **Methods** (one precise definition) + **short Limitations reminder** |
| **B** | Layer C downstream of image-level score | **System overview / Fig. 1** + **Results or Discussion** (not every subsection) |
| **C** | Figures are qualitative (≠ quantitative benchmark) | **Caption** + **brief Results/Discussion once** |
| **D** | Human-reviewed test partition reserved | **Experimental Protocol** + **Limitations** |
| **E** | Certification / flight-qualification envelope | **Hardware scope and/or Limitations once** — no repeated “not flight-qualified” stacks |

---

## Rewrite priority (for follow-on TeX pass)

1. **Discussion + Limitations + Methods depth/fusion/Layer stacks** (highest A density).  
2. **Fig. 1–4 captions + Table 1 cells**.  
3. **Experiments / Results** reserved-test and Layer B characterization.  
4. **Intro / Related Work** certification + depth.  
5. **Abstract** metric-distance parenthetical only.  
6. **Conclusion** fixed-coefficient vs entropy framing.  
7. Theme **A–E** dedupe sweep + PDF negation QA.
