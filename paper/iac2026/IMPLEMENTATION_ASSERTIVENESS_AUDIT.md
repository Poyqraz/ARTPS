# Implementation assertiveness audit

Reviewer-visible `may` / `might` / `could` / `optional` / `where available` after the primary-evaluation completeness pass. Genuine optionals stay; hedging of performed work does not.

| Location | Phrase | Keep? | Reason |
| --- | --- | --- | --- |
| Fig. 1 / Methods overview | optional enhancement; optional classifier | YES | Implemented slots; enhancement disabled in reported experiments |
| Methods 3.2 | reconstruction residual may under-emphasize | YES | Cue complementarity, not a claim that a stage was skipped |
| Methods 3.1 | Optional enhancement may precede | YES | Preprocessing path, not a detection metric |
| Methods localization | optional feature descriptor | YES | Diversity descriptor is not required for the image-level score |
| Methods scoring | optional PaDiM and PatchCore responses | YES | Disabled in human-reviewed validation; baseline only |
| Methods scoring | recall-support pooling may raise \(F_r\) | YES | Optional pooling before the candidate equations |
| Methods curiosity | optional combined-map | YES | Ranking term, not image-level score |
| Methods suppression | invalidated where those signals are available | YES | Overlay/telemetry masks are conditional on signal presence |
| Fig. 2/4 captions; Discussion | where available (support geometry) | YES | Conditional overlay; not a claim that every candidate has a footprint |
| Discussion | future challenge set may be useful | YES | Future work |
| Methods ranking | enter / reduces / retains | ASSERTED | Implemented Layer C sequence |
| Methods fail-closed | returns no-selection when evidence is insufficient | ASSERTED | Specified runtime behaviour |

No `might` / `could` / `appears to` / `was intended to` / `was planned to` in reviewer-visible section TeX.

Do not imply that unevaluated items (Jetson, full-pipeline edge timing, metric depth, flight qualification) were measured.
