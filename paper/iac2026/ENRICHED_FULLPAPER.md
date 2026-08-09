# Enriched IAC full-paper variant

Status: **candidate**. The 7-page freeze commit remains the conservative rollback.

- PR #37 added Fig.~2 (current-code qualitative) plus moderate Methods / Protocol prose.
- Context pass: evaluation-track table + suppression / ranking explanation; no new
  quantitative experiment, no test split, no Jetson, no historical metric changes.
- Cue-decomposition Fig.~3: same non-test sample as Fig.~2; frozen fusion cues only;
  provenance [`figures/FIG_CUE_DECOMPOSITION_PROVENANCE.md`](figures/FIG_CUE_DECOMPOSITION_PROVENANCE.md).
- Figure provenance: [`figures/FIG_QUALITATIVE_PROVENANCE.md`](figures/FIG_QUALITATIVE_PROVENANCE.md).
- Scientific freeze snapshot in [`SUBMISSION_FREEZE.md`](SUBMISSION_FREEZE.md) is unchanged
  except `enriched_fullpaper_candidate: true`, `enriched_context_pass: true`,
  `cue_decomposition_figure: true`, and `scientific_definition_pass_complete: true`.
