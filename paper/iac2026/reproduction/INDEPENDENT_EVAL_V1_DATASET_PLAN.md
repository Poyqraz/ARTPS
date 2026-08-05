# Independent evaluation v1 — dataset acquisition plan

**Protocol:** [INDEPENDENT_EVALUATION_PROTOCOL.md](INDEPENDENT_EVALUATION_PROTOCOL.md)  
**Annotation:** [INDEPENDENT_EVAL_V1_ANNOTATION_GUIDE.md](INDEPENDENT_EVAL_V1_ANNOTATION_GUIDE.md)

This plan pins **how** a labeled, SHA-pinned set will be built. It does **not**
add dataset rows in this PR. Accepted-abstract aggregates
(2847 / 1247 / 892 / 708) are **not** quotas.

---

## Sources

| Mission | Instrument | Access |
|---------|------------|--------|
| Curiosity | Mastcam | NASA PDS / planetary data APIs or documented archive mirrors |
| Perseverance | Mastcam-Z | Same |

Document the exact download tool, query, and date for each acquisition batch in
the run notes. Prefer stable `product_id` / `source_id` / `source_url`.

## Inclusion criteria

- RGB surface images with usable focus/exposure for annotation under the guide
- Traceable product identity (`product_id`, `source_id`, preferably `source_url`)
- Byte SHA available for raw and (if used) derived files (`raw_sha256` ≠ confuse with `derived_sha256`)

## Exclusion criteria

- Rover-dominated frames where no independent surface target remains after masks
- Overlay / border / telemetry-corrupted frames
- Near-duplicates / crops of already included samples
- Frames failing minimum quality (severe blur, extreme underexposure/overexposure)
- Any frame selected because of a model score (forbidden)

## Sol / date scope

Choose an explicit Sol (or Earth-date) window **before** labeling begins and
record it in the protocol bump notes. Do not expand the window after seeing
test metrics.

## Sample selection procedure

1. Enumerate candidate products from the pinned source queries (no model scores).
2. Deduplicate by content hash and near-duplicate / crop rules.
3. Filter rover-body / overlay / quality rejects.
4. Sample for annotation without using ARTPS or baseline scores.
5. Only after labels exist, set `split_ratios` in the protocol lock and bump
   `protocol_version` — never invent ratios from historical manuscript counts.

## Duplicate / crop detection

- Exact: identical `sha256` / `raw_sha256`
- Near: shared `duplicate_group_id` policy (perceptual or metadata grouping)
- Same `sequence_id` / `scene_group_id` must not leak across splits

## Reporting requirements (before claiming a finished set)

- Mission × instrument counts
- Class balance overall and by split (after split freeze)
- Exclusion reason histogram
- Annotator / adjudication stats

## Target sample size

Determine N from annotation budget and statistical plan needs **after** seeing
candidate pool size — not by targeting 2847/1247/892/708. Record the decision
in the protocol version changelog when ratios are unlocked.
