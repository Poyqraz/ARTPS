# Shadow Dual-Track + ESRGAN Design

**Date:** 2026-07-20  
**Status:** Approved (brainstorm)

## Decisions

- Goals: visibility (A) + analysis recall (B), parallel — not sequential dual RGB
- Architecture: single shared RGB + dual logic
- ESRGAN: optional on shared Mars-safe enhance (feeds analysis when enabled)
- Shadow lift: classical mask gamma→CLAHE default; optional depth-gate
- Approach: lean dual-logic (no DL shadow removal / LiDAR / Silva full port)

## Architecture

Single Shared RGB from `enhance_image_auto` → `enhanced_image_for_analysis`.

- **Track A (visibility):** mask-guided local gamma→CLAHE preview; does not write analysis session
- **Track B (analysis):** `object_in_shadow` gate softens `alpha_shad` + small soft recall
- **ESRGAN:** denoise → SR x2 → cubic → CLAHE → gamma → unsharp; CUDA half

## Track A

1. Soft shadow mask (RGB-only or `compute_shadow_like` with depth)
2. Default lift: masked gamma (~120) + CLAHE clip 1.5; lerp with mask
3. Optional depth-gate: weaken lift where depth edge is high
4. UI before/after + mask overlay only

## Track B

1. `object_in_shadow` = dark × max(image_edge, depth_edge)
2. `alpha_map = alpha_shad * (1 - beta * object_gate)`
3. Soft recall via small `gamma_recall * object_gate`
4. Sidebar `beta_shadow_obj` default 0.5; `beta=0` ≈ legacy

## ESRGAN

- Reorder after denoise; `half=torch.cuda.is_available()`
- OOM: retry tile 200 then fallback
- Default off; risk caption unchanged

## Errors / tests / out of scope

- Depth missing → RGB-only Track A; Track B depth_edge=0 path via flat depth or RGB edges only
- Tests: visibility lift, gate suppression, ESRGAN step order
- Out of scope: DL shadow removal, ROI-only ESRGAN, GFPGAN, second analysis RGB
