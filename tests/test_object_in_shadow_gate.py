"""Track B: object-in-shadow gate and gated suppression."""
from __future__ import annotations

import numpy as np

from src.core.false_positive_masks import (
    apply_gated_shadow_suppression,
    compute_object_in_shadow,
    compute_shadow_like,
)


def test_object_in_shadow_low_on_flat_dark() -> None:
    rgb = np.full((64, 64, 3), 0.12, np.float32)
    depth = np.full((64, 64), 0.4, np.float32)
    gate = compute_object_in_shadow(rgb, depth)
    assert float(gate.mean()) < 0.15


def test_object_in_shadow_high_on_dark_with_edges() -> None:
    rgb = np.full((64, 64, 3), 0.12, np.float32)
    depth = np.full((64, 64), 0.4, np.float32)
    # Checker / step edges in both RGB and depth
    rgb[:, 32:] = 0.35
    depth[:, 32:] = 0.9
    gate = compute_object_in_shadow(rgb, depth)
    assert float(gate[:, 28:36].mean()) > float(gate[:, 8:16].mean())
    assert float(gate[:, 28:36].mean()) > 0.15


def test_gated_suppression_weaker_than_flat_when_beta_one() -> None:
    combined = np.ones((32, 32), np.float32)
    shadow = np.ones((32, 32), np.float32) * 0.8
    # Left: flat shadow (gate=0); right: object-in-shadow (gate=1)
    gate = np.zeros((32, 32), np.float32)
    gate[:, 16:] = 1.0

    out = apply_gated_shadow_suppression(
        combined,
        shadow,
        gate,
        alpha_shad=0.65,
        beta=1.0,
        gamma_recall=0.0,
    )
    # Right (gated) should be suppressed less than left (flat)
    assert float(out[:, 16:].mean()) > float(out[:, :16].mean())


def test_beta_zero_matches_legacy_alpha() -> None:
    combined = np.ones((16, 16), np.float32)
    shadow = np.full((16, 16), 0.5, np.float32)
    gate = np.ones((16, 16), np.float32)
    out = apply_gated_shadow_suppression(
        combined, shadow, gate, alpha_shad=0.65, beta=0.0, gamma_recall=0.0
    )
    expected = 1.0 - 0.65 * 0.5
    assert abs(float(out.mean()) - expected) < 1e-5
    # shadow_like still used elsewhere
    assert compute_shadow_like(
        np.full((16, 16, 3), 0.1, np.float32),
        np.full((16, 16), 0.5, np.float32),
    ).mean() > 0.3
