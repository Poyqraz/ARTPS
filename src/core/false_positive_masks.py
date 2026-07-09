from __future__ import annotations

import cv2
import numpy as np


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _gray_depth_edges(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    image_grad = _normalize(np.hypot(grad_x, grad_y))

    depth = depth.astype(np.float32, copy=False)
    depth_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    depth_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_edge = _normalize(np.hypot(depth_x, depth_y))

    return gray, image_grad, depth_edge


def compute_shadow_like(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Koyu, düz ve derinlikte keskin olmayan bölgeleri öne çıkar."""
    _, image_grad, depth_edge = _gray_depth_edges(rgb, depth)
    hsv = cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    dark = 1.0 - hsv[..., 2]
    shadow_like = np.clip(dark * (1.0 - image_grad) * (1.0 - depth_edge), 0.0, 1.0)
    return cv2.GaussianBlur(shadow_like, (5, 5), 0)


def compute_rover_body_mask(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    bottom_frac: float = 0.28,
    near_thresh: float = 0.45,
) -> np.ndarray:
    """Alt görüş alanındaki yakın ve pürüzsüz rover gövdesini işaretle."""
    height, width = depth.shape[:2]
    _, _, depth_edge = _gray_depth_edges(rgb, depth)
    hsv = cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    low_saturation = 1.0 - hsv[..., 1]
    proximity = _normalize(1.0 - depth.astype(np.float32, copy=False))

    band = np.zeros((height, width), dtype=np.float32)
    start_row = int(height * (1.0 - bottom_frac))
    band[start_row:, :] = 1.0

    rover_body = band
    rover_body *= (proximity > near_thresh).astype(np.float32)
    rover_body *= (1.0 - depth_edge)
    rover_body *= (0.5 + 0.5 * low_saturation)
    return cv2.GaussianBlur(np.clip(rover_body, 0.0, 1.0), (5, 5), 0)


def compute_cast_shadow_mask(
    shadow_like: np.ndarray,
    rover_body: np.ndarray,
    *,
    dilate_k: int = 15,
) -> np.ndarray:
    """Rover gövdesine komşu gölgeyi araç gölgesi kabul et."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    rover_zone = cv2.dilate((rover_body > 0.35).astype(np.uint8), kernel, iterations=1).astype(np.float32)
    return np.clip(shadow_like * rover_zone, 0.0, 1.0)


def compute_horizon_mask(
    depth: np.ndarray,
    *,
    depth_thresh: float = 0.8,
    edge_thresh: float = 0.05,
) -> np.ndarray:
    """Uzak ve derinlikte düz ufuk alanlarını işaretle."""
    depth = depth.astype(np.float32, copy=False)
    depth_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    depth_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_edge = _normalize(np.hypot(depth_x, depth_y))
    return ((depth > depth_thresh) & (depth_edge < edge_thresh)).astype(np.float32)


def apply_fp_suppression(
    combined: np.ndarray,
    *,
    shadow_like: np.ndarray | None = None,
    rover_body: np.ndarray | None = None,
    cast_shadow: np.ndarray | None = None,
    horizon: np.ndarray | None = None,
    alpha_shad: float = 0.65,
    alpha_rover: float = 0.85,
    alpha_cast: float = 0.75,
    alpha_horizon: float = 0.55,
) -> np.ndarray:
    """False-positive maskelerini çarpımsal olarak uygula."""
    output = combined.astype(np.float32, copy=True)
    if shadow_like is not None:
        output *= 1.0 - alpha_shad * shadow_like
    if cast_shadow is not None:
        output *= 1.0 - alpha_cast * cast_shadow
    if rover_body is not None:
        output *= 1.0 - alpha_rover * rover_body
    if horizon is not None:
        output *= 1.0 - alpha_horizon * horizon
    return np.clip(output, 0.0, 1.0)


__all__ = [
    "apply_fp_suppression",
    "compute_cast_shadow_mask",
    "compute_horizon_mask",
    "compute_rover_body_mask",
    "compute_shadow_like",
]
