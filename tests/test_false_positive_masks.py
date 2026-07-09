import numpy as np

from src.core.false_positive_masks import (
    apply_fp_suppression,
    compute_boundary_shadow_mask,
    compute_rover_body_mask,
    compute_shadow_like,
)


def test_boundary_shadow_keeps_only_border_connected_dark_regions():
    rgb = np.full((64, 64, 3), 0.65, np.float32)
    depth = np.full((64, 64), 0.5, np.float32)

    rgb[44:, :18] = 0.08
    rgb[28:36, 30:38] = 0.05

    mask = compute_boundary_shadow_mask(rgb, depth)

    assert mask[48:60, 2:14].mean() > 0.7
    assert mask[29:35, 31:37].mean() < 0.2


def test_shadow_like_high_on_dark_flat():
    rgb = np.full((64, 64, 3), 0.15, np.float32)
    depth = np.full((64, 64), 0.4, np.float32)

    mask = compute_shadow_like(rgb, depth)

    assert mask.mean() > 0.4


def test_rover_body_mask_keeps_border_near_blob_drops_center_and_sky():
    rng = np.random.default_rng(0)
    depth = 0.5 + rng.normal(0.0, 0.02, size=(96, 96)).astype(np.float32)

    # Sinira bagli, dokulu yakin rover parcasi (buyuk depth sapmasi) - saga yaslanik
    depth[:, 80:] = 0.9
    # Merkezdeki kucuk tas (kucuk sapma, sinira degmiyor) - korunmali
    depth[44:52, 44:52] = 0.55
    # Ust-sol gokyuzu bandi (buyuk sapma ama yalniz ust/sol sinir) - elenmeli
    depth[:10, :40] = 0.05

    mask = compute_rover_body_mask(depth)

    assert mask[30:70, 82:94].mean() > 0.7
    assert mask[44:52, 44:52].mean() < 0.2
    assert mask[2:8, 5:35].mean() < 0.2


def test_apply_fp_suppression_reduces_rover_body_region():
    combined = np.ones((32, 32), np.float32)
    rover = np.zeros((32, 32), np.float32)
    rover[:, 24:] = 1.0

    output = apply_fp_suppression(
        combined,
        rover_body=rover,
        alpha_rover=0.9,
    )

    assert output[:, 24:].mean() < output[:, :24].mean()


def test_apply_fp_suppression_reduces_boundary_shadow_region():
    combined = np.ones((32, 32), np.float32)
    boundary = np.zeros((32, 32), np.float32)
    boundary[:, :16] = 1.0

    output = apply_fp_suppression(
        combined,
        boundary_shadow=boundary,
        alpha_boundary=0.85,
    )

    assert output[:, :16].mean() < output[:, 16:].mean()
