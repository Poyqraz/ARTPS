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


def _as_u8_rgb(rgb: np.ndarray) -> np.ndarray:
    """Accept float[0,1] or uint8 RGB."""
    if rgb.dtype == np.uint8:
        return rgb
    return _to_uint8_rgb(rgb)


def image_depth_edges(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(_as_u8_rgb(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
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


# Back-compat alias
_image_depth_edges = image_depth_edges


def _darkness(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(_as_u8_rgb(rgb), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    return 1.0 - hsv[..., 2]


def compute_shadow_mask(rgb: np.ndarray, depth: np.ndarray | None = None) -> np.ndarray:
    """Soft flat-shadow mask; RGB-only when depth is None."""
    dark = _darkness(rgb)
    # Soft gate: mid/bright flats should not count as shadow (V high → dark low)
    dark_w = np.clip((dark - 0.35) / 0.30, 0.0, 1.0)
    if depth is None:
        gray = cv2.cvtColor(_as_u8_rgb(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        image_edge = _normalize(
            np.hypot(
                cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
            )
        )
        shadow = np.clip(dark * (1.0 - image_edge) * dark_w, 0.0, 1.0)
    else:
        image_edge, depth_edge = image_depth_edges(rgb, depth)
        shadow = np.clip(dark * (1.0 - image_edge) * (1.0 - depth_edge) * dark_w, 0.0, 1.0)
    return cv2.GaussianBlur(shadow, (5, 5), 0)


def compute_shadow_like(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Koyu, düz ve derinlikte keskin olmayan bölgeleri öne çıkar."""
    return compute_shadow_mask(rgb, depth)


def compute_object_in_shadow(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Dark regions that also have image/depth structure (rock base objects)."""
    image_edge, depth_edge = image_depth_edges(rgb, depth)
    dark = _darkness(rgb)
    structure = np.maximum(image_edge, depth_edge)
    gate = np.clip(dark * structure, 0.0, 1.0)
    return cv2.GaussianBlur(gate, (5, 5), 0)


def apply_gated_shadow_suppression(
    combined: np.ndarray,
    shadow_like: np.ndarray,
    object_gate: np.ndarray,
    *,
    alpha_shad: float = 0.65,
    beta: float = 0.5,
    gamma_recall: float = 0.08,
) -> np.ndarray:
    """Suppress flat shadows; soften suppression / soft-recall where object_gate is high.

    beta=0 → legacy: combined *= (1 - alpha_shad * shadow_like).
    """
    beta = float(np.clip(beta, 0.0, 1.0))
    alpha_map = float(alpha_shad) * (1.0 - beta * object_gate.astype(np.float32))
    out = combined.astype(np.float32, copy=True)
    out *= 1.0 - alpha_map * shadow_like.astype(np.float32)
    # Soft recall of existing signal in structured-shadow regions
    out = out + float(gamma_recall) * object_gate.astype(np.float32) * combined.astype(np.float32)
    return np.clip(out, 0.0, 1.0)


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


def compute_rover_body_mask(
    depth: np.ndarray,
    *,
    deviation_k: float = 2.5,
    min_area_frac: float = 0.01,
    top_exclude_frac: float = 0.25,
) -> np.ndarray:
    """Sinira bagli araç gövdesi (tekerlek/kol) bölgelerini yakala.

    Depth isaret yönünden bagimsizdir: merkez arazi derinligine göre robust
    MAD sapmasi kullanir. Rover parcalari (yakin) ve gökyüzü (uzak) yüksek
    sapma verir; sadece sol/sag/alt sinira bagli ve ust bantta olmayan
    bilesenler tutulur (gökyüzü elenir).
    """
    depth = depth.astype(np.float32, copy=False)
    height, width = depth.shape[:2]
    cy0, cy1 = int(0.3 * height), int(0.7 * height)
    cx0, cx1 = int(0.3 * width), int(0.7 * width)
    central = depth[cy0:cy1, cx0:cx1]
    ref = float(np.median(central))
    mad = float(np.median(np.abs(central - ref))) + 1e-6
    deviation = np.abs(depth - ref) / mad

    binary = (deviation > deviation_k).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(1, int(height * width * min_area_frac))
    top_limit = top_exclude_frac * height
    keep = np.zeros(depth.shape, dtype=np.float32)
    for label in range(1, labels_count):
        x, y, w, h, area = stats[label]
        touches_side_or_bottom = x == 0 or x + w >= width or y + h >= height
        if touches_side_or_bottom and area >= min_area and centroids[label][1] > top_limit:
            keep[labels == label] = 1.0
    return cv2.GaussianBlur(keep, (5, 5), 0)


def apply_fp_suppression(
    combined: np.ndarray,
    *,
    shadow_like: np.ndarray | None = None,
    boundary_shadow: np.ndarray | None = None,
    rover_body: np.ndarray | None = None,
    alpha_shad: float = 0.65,
    alpha_boundary: float = 0.85,
    alpha_rover: float = 0.9,
) -> np.ndarray:
    output = combined.astype(np.float32, copy=True)
    if shadow_like is not None:
        output *= 1.0 - alpha_shad * shadow_like
    if boundary_shadow is not None:
        output *= 1.0 - alpha_boundary * boundary_shadow
    if rover_body is not None:
        output *= 1.0 - alpha_rover * rover_body
    return np.clip(output, 0.0, 1.0)


__all__ = [
    "apply_fp_suppression",
    "apply_gated_shadow_suppression",
    "compute_boundary_shadow_mask",
    "compute_object_in_shadow",
    "compute_rover_body_mask",
    "compute_shadow_like",
    "compute_shadow_mask",
    "image_depth_edges",
]
