"""Track A: shadow visibility lift for human preview (does not feed analysis)."""
from __future__ import annotations

import cv2
import numpy as np

from src.core.false_positive_masks import compute_shadow_mask, image_depth_edges


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    if rgb.dtype in (np.float32, np.float64) and float(np.nanmax(rgb)) <= 1.5:
        return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _clahe_lab(rgb_u8: np.ndarray, clip: float = 1.5) -> np.ndarray:
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(L), A, B]), cv2.COLOR_LAB2RGB)


def lift_shadow_visibility(
    rgb: np.ndarray,
    depth: np.ndarray | None = None,
    *,
    use_depth_gate: bool = False,
    target_mean: float = 120.0,
    clahe_clip: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask-guided gamma→CLAHE lift for preview.

    Returns (lifted_uint8_rgb, soft_mask float32 in [0,1]).
    Does not mutate analysis session — callers must not assign to analysis input.
    """
    orig = _as_uint8_rgb(rgb)
    mask = compute_shadow_mask(orig, depth)

    if use_depth_gate and depth is not None:
        _, depth_edge = image_depth_edges(orig, depth)
        mask = np.clip(mask * (1.0 - depth_edge), 0.0, 1.0)

    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Mean only where mask is strong AND pixels are actually dark
    heavy = (mask > 0.35) & (gray < 90.0)
    if np.any(heavy):
        shadow_mean = float(gray[heavy].mean()) + 1e-6
    else:
        shadow_mean = float((gray * mask).sum() / (float(mask.sum()) + 1e-6)) + 1e-6
    # Gamma from shadow ROI only so bright flats are not the lift driver
    gamma = float(
        np.clip(
            np.log(target_mean / 255.0 + 1e-6) / np.log(shadow_mean / 255.0 + 1e-6),
            0.4,
            2.0,
        )
    )
    gamma_img = np.clip((orig.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    lifted = _clahe_lab(gamma_img, clip=clahe_clip)
    m = mask[..., None].astype(np.float32)
    out = np.clip(
        orig.astype(np.float32) * (1.0 - m) + lifted.astype(np.float32) * m,
        0,
        255,
    ).astype(np.uint8)
    return out, mask.astype(np.float32)
