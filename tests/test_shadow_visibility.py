"""Track A shadow visibility lift tests."""
from __future__ import annotations

import numpy as np

from src.utils.shadow_visibility import lift_shadow_visibility


def test_lift_raises_dark_roi_preserves_bright() -> None:
    rgb = np.full((64, 64, 3), 180, dtype=np.uint8)
    rgb[40:60, 40:60] = 25
    out, mask = lift_shadow_visibility(rgb, depth=None, use_depth_gate=False)

    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    assert mask.shape == (64, 64)

    dark_before = float(rgb[40:60, 40:60].mean())
    dark_after = float(out[40:60, 40:60].mean())
    bright_before = float(rgb[0:20, 0:20].mean())
    bright_after = float(out[0:20, 0:20].mean())

    assert dark_after > dark_before
    assert abs(bright_after - bright_before) / max(bright_before, 1.0) < 0.05


def test_depth_gate_weakens_mask_on_depth_edge() -> None:
    rgb = np.full((64, 64, 3), 20, dtype=np.uint8)
    depth = np.full((64, 64), 0.5, dtype=np.float32)
    # Strong depth edge on left half
    depth[:, :32] = 0.1
    depth[:, 32:] = 0.9

    _, mask_gated = lift_shadow_visibility(rgb, depth, use_depth_gate=True)
    _, mask_flat = lift_shadow_visibility(rgb, depth, use_depth_gate=False)

    # Edge band around column 32 should be weaker under depth gate
    edge_band = mask_gated[:, 28:36].mean()
    flat_band = mask_flat[:, 28:36].mean()
    assert edge_band < flat_band
