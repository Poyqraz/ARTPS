"""Canonical size/distance features and 2x2 proposal policy (no UI)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SizeDistanceBand = Literal[
    "near_small",
    "near_large",
    "far_small",
    "far_large",
    "mid",
]

# Fixed apparent_size compression (testable, no image-global percentile)
_APPARENT_K = 40.0
_APPARENT_REF = 0.05 / (1e-3 + 0.5**2)  # area 0.05 at far=0.5


@dataclass(frozen=True)
class SizeDistanceFeatures:
    relative_far: float
    apparent_size: float
    area_ratio: float
    metric_proxy: float | None
    metric_size: float | None
    band: SizeDistanceBand


def estimate_depth_scale_m(depth_map: np.ndarray) -> float | None:
    """ponytail: calibration hook; returns None until scale-aware depth exists."""
    _ = depth_map
    return None


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _classify_band(relative_far: float, apparent_size: float) -> SizeDistanceBand:
    far = float(relative_far)
    app = float(apparent_size)
    if 0.4 <= far < 0.55 or 0.35 <= app < 0.45:
        # Mid band if either axis is mid — unless both extremes agree
        if far < 0.4 and app < 0.35:
            return "near_small"
        if far < 0.4 and app >= 0.45:
            return "near_large"
        if far >= 0.55 and app < 0.35:
            return "far_small"
        if far >= 0.55 and app >= 0.45:
            return "far_large"
        return "mid"
    if far < 0.4 and app < 0.35:
        return "near_small"
    if far < 0.4 and app >= 0.45:
        return "near_large"
    if far >= 0.55 and app < 0.35:
        return "far_small"
    if far >= 0.55 and app >= 0.45:
        return "far_large"
    return "mid"


def compute_size_distance_features(
    *,
    w: int,
    h: int,
    img_hw: tuple[int, int],
    depth_crop: np.ndarray | None = None,
    proximity_crop: np.ndarray | None = None,
    depth_sign: float = 1.0,
    depth_scale_m: float | None = None,
    depth_span: float | None = None,
) -> SizeDistanceFeatures:
    """Build size/distance features for one bbox.

    Prefer proximity_crop when provided: relative_far = 1 - median(proximity)
    so the sign matches app proximity_w = normalize(1 - depth).
    """
    H, W = int(img_hw[0]), int(img_hw[1])
    area_ratio = float((max(1, int(w)) * max(1, int(h))) / max(1.0, float(H * W)))

    if proximity_crop is not None and np.size(proximity_crop) > 0:
        prox = np.asarray(proximity_crop, dtype=np.float32)
        relative_far = _clip01(1.0 - float(np.median(prox)))
    elif depth_crop is not None and np.size(depth_crop) > 0:
        d = np.asarray(depth_crop, dtype=np.float32)
        dmin, dmax = float(np.min(d)), float(np.max(d))
        if dmax - dmin < 1e-8:
            relative_far = 0.5
        else:
            dn = (d - dmin) / (dmax - dmin)
            if float(depth_sign) < 0:
                dn = 1.0 - dn
            relative_far = _clip01(float(np.median(dn)))
    else:
        relative_far = 0.5

    apparent_raw = area_ratio / (1e-3 + relative_far**2)
    apparent_size = _clip01(
        float(np.log1p(_APPARENT_K * apparent_raw) / np.log1p(_APPARENT_K * _APPARENT_REF))
    )

    metric_proxy: float | None = None
    if depth_span is not None:
        metric_proxy = _clip01(apparent_size * (0.5 + 0.5 * float(np.clip(depth_span, 0.0, 1.0))))

    metric_size: float | None = None
    if depth_scale_m is not None and metric_proxy is not None:
        metric_size = float(metric_proxy) * float(depth_scale_m)

    band = _classify_band(relative_far, apparent_size)
    return SizeDistanceFeatures(
        relative_far=relative_far,
        apparent_size=apparent_size,
        area_ratio=area_ratio,
        metric_proxy=metric_proxy,
        metric_size=metric_size,
        band=band,
    )


def area_min_scale(feat: SizeDistanceFeatures) -> float:
    """Multiplier on contour area_min (lower = easier to keep small blobs)."""
    if feat.band == "far_small":
        return 0.35
    if feat.band == "near_large":
        return 1.0
    if feat.band == "near_small":
        return 0.70
    if feat.band == "far_large":
        return 0.55
    # mid / legacy-like: same shape as old (0.35 + 0.65*(1-far))
    return float(0.35 + 0.65 * (1.0 - feat.relative_far))


def edge_min_scale(feat: SizeDistanceFeatures) -> float:
    """Multiplier on img/depth edge minima (lower = looser for far-small)."""
    if feat.band == "far_small":
        return 0.55
    if feat.band == "near_large":
        return 1.0
    if feat.band == "near_small":
        return 0.85
    if feat.band == "far_large":
        return 0.75
    return float(1.0 - 0.35 * feat.relative_far)


def shadow_cut_delta(feat: SizeDistanceFeatures) -> float:
    """Added to shadow_cut / similar cuts (far → slightly harder FP cuts)."""
    return float(0.10 * feat.relative_far)


def merge_bridge_floor(
    feat_a: SizeDistanceFeatures,
    feat_b: SizeDistanceFeatures,
    base: float,
) -> float:
    """Raise required heatmap bridge for near-large × near-large merges."""
    if feat_a.band == "near_large" and feat_b.band == "near_large":
        return float(max(base, 0.12, base * 2.0))
    return float(base)


def should_reject_field_scale(feat: SizeDistanceFeatures, support: float) -> bool:
    """Reject field-scale near-large / huge boxes with weak anomaly support."""
    if feat.band == "near_large" and feat.area_ratio >= 0.08 and float(support) < 0.04:
        return True
    if feat.area_ratio >= 0.12 and feat.apparent_size >= 0.70 and float(support) < 0.03:
        return True
    return False


def features_from_det_fields(det: dict) -> SizeDistanceFeatures | None:
    """Rebuild features from annotated det fields if present."""
    if "relative_far" not in det or "apparent_size" not in det:
        return None
    return SizeDistanceFeatures(
        relative_far=float(det["relative_far"]),
        apparent_size=float(det["apparent_size"]),
        area_ratio=float(det.get("area_ratio", 0.0)),
        metric_proxy=det.get("metric_proxy"),
        metric_size=det.get("metric_size"),
        band=str(det.get("size_distance_band", "mid")),  # type: ignore[arg-type]
    )
