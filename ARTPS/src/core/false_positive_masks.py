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


def _image_depth_edges(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    image_edge = _normalize(
        np.hypot(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
    )
    depth = depth.astype(np.float32, copy=False)
    depth_edge = _normalize(
        np.hypot(
            cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3),
        )
    )
    return image_edge, depth_edge


def compute_shadow_like(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Koyu, düz ve derinlikte keskin olmayan bölgeleri öne çıkar."""
    image_edge, depth_edge = _image_depth_edges(rgb, depth)
    hsv = cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    dark = 1.0 - hsv[..., 2]
    shadow_like = np.clip(dark * (1.0 - image_edge) * (1.0 - depth_edge), 0.0, 1.0)
    return cv2.GaussianBlur(shadow_like, (5, 5), 0)


def compute_boundary_shadow_mask(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    min_area_frac: float = 0.002,
) -> np.ndarray:
    """Sadece görüntü sınırına bağlı gölge/gövde bileşenlerini tut."""
    shadow = compute_shadow_like(rgb, depth)
    binary = (shadow > 0.35).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    min_area = max(1, int(height * width * min_area_frac))
    keep = np.zeros_like(binary, dtype=np.float32)
    for label in range(1, labels_count):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if touches_border and area >= min_area:
            keep[labels == label] = 1.0
    return cv2.GaussianBlur(keep, (5, 5), 0)


def apply_fp_suppression(
    combined: np.ndarray,
    *,
    shadow_like: np.ndarray | None = None,
    boundary_shadow: np.ndarray | None = None,
    alpha_shad: float = 0.65,
    alpha_boundary: float = 0.85,
) -> np.ndarray:
    output = combined.astype(np.float32, copy=True)
    if shadow_like is not None:
        output *= 1.0 - alpha_shad * shadow_like
    if boundary_shadow is not None:
        output *= 1.0 - alpha_boundary * boundary_shadow
    return np.clip(output, 0.0, 1.0)


__all__ = [
    "apply_fp_suppression",
    "compute_boundary_shadow_mask",
    "compute_shadow_like",
]
