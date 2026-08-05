# Independent evaluation v1 — annotation guide

**Annotation version:** `independent_eval_v1`

**Protocol:** [INDEPENDENT_EVALUATION_PROTOCOL.md](INDEPENDENT_EVALUATION_PROTOCOL.md)

This guide defines human labels for the **current reproducible evaluation**. It does
**not** claim mineral detection, compositional ID, or equivalence to unrecovered
historical C05/C06 ground truth.

---

## A. Positive label (`binary_label=1`)

An image is positive when it contains **at least one real surface object or region**
that is scientifically worth examining and is visually distinct from its immediate
surroundings under this guide (rocks with unusual morphology/texture/context,
distinct deposits, or other science-interest targets that a rover operator would
flag for follow-up imaging).

## B. Negative label (`binary_label=0`)

An image is negative when, under this guide, it does **not** contain a real
anomaly / science-interest target as defined above (typical terrain without a
distinct target worth follow-up).

## C. Must not be counted as positive (alone)

- Rover body / wheel / arm / hardware
- Image border / frame edge artifacts
- Telemetry / text / UI overlay
- Compression artefacts
- Dead / hot pixels
- Shadow boundary alone (without a distinct target)
- Exposure / specular artefact alone
- Blur / noise alone
- Duplicate or cropped version of another already-labeled sample

## D. Ambiguous class

Use one of:

- `uncertain` + `inclusion_status=uncertain` (exclude from primary eval), or
- `exclude_from_primary_eval` via `inclusion_status=excluded` with
  `exclusion_reason` filled, or
- Pre-defined adjudication workflow ending in `adjudication_status=resolved`

Primary evaluation accepts only:

`inclusion_status=included` AND `adjudication_status=resolved` AND
`annotation_version=independent_eval_v1`.

## E. Annotation process

1. Model scores must **not** be shown to annotators.
2. Annotators must label **without** split membership information.
3. Prefer **at least two** independent annotators.
4. Disagreements resolved by a third / adjudicating review.
5. Record `label_source`, `annotator_count`, `adjudication_status`,
   `label_confidence`.
6. Guide or label-rule changes require a new `annotation_version` (do not silently
   rewrite history).

## F. Edge cases (document in `notes` when relevant)

- Small / distant target
- Rock partially in shadow
- Object near rover hardware
- Multiple candidate targets in one frame
- Geological layering / texture without a discrete target
- Low image quality
- Near-duplicate or crop of another sample

When unsure whether a case is positive, prefer `uncertain` / exclude from primary
eval rather than inventing a positive.
