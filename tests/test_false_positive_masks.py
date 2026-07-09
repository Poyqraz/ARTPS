import numpy as np

from src.core.false_positive_masks import (
    apply_fp_suppression,
    compute_cast_shadow_mask,
    compute_horizon_mask,
    compute_rover_body_mask,
    compute_shadow_like,
)


def _flat(h: int = 64, w: int = 64, gray: float = 0.5):
    rgb = np.full((h, w, 3), gray, np.float32)
    depth = np.full((h, w), 0.5, np.float32)
    return rgb, depth


def test_shadow_like_high_on_dark_flat():
    rgb, depth = _flat(gray=0.15)
    depth[:] = 0.4
    mask = compute_shadow_like(rgb, depth)
    assert mask.mean() > 0.4


def test_shadow_like_low_on_bright_textured():
    rgb, depth = _flat(gray=0.8)
    rgb[::2, ::2] = 0.2
    mask = compute_shadow_like(rgb, depth)
    assert mask.mean() < 0.35


def test_rover_body_bottom_near_smooth():
    rgb, depth = _flat(gray=0.35)
    depth[48:, :] = 0.1
    depth[:48, :] = 0.7
    mask = compute_rover_body_mask(rgb, depth, bottom_frac=0.25)
    assert mask[48:, :].mean() > mask[:48, :].mean()


def test_horizon_mask_far_flat():
    _, depth = _flat()
    depth[:20, :] = 0.95
    depth[20:, :] = 0.3
    mask = compute_horizon_mask(depth)
    assert mask[:20, :].mean() > 0.5
    assert mask[20:, :].mean() < 0.2


def test_apply_fp_suppression_reduces_shadow_region():
    combined = np.ones((32, 32), np.float32)
    shadow = np.zeros((32, 32), np.float32)
    shadow[:, :16] = 1.0
    output = apply_fp_suppression(combined, shadow_like=shadow, alpha_shad=0.8)
    assert output[:, :16].mean() < output[:, 16:].mean()


def test_cast_shadow_tracks_rover_adjacency():
    shadow = np.zeros((32, 32), np.float32)
    shadow[:, :20] = 1.0
    rover = np.zeros((32, 32), np.float32)
    rover[:, 8:16] = 1.0
    mask = compute_cast_shadow_mask(shadow, rover, dilate_k=9)
    assert mask[:, :20].mean() > mask[:, 24:].mean()
