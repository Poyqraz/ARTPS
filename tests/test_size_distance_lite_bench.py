"""Sentetik 2x2 size/distance lite bench — policy davranışı (CI-safe, AE/DPT yok)."""
from __future__ import annotations

import numpy as np

from app import _should_keep_detection, _should_merge_proposals
from src.core.size_distance import (
    area_min_scale,
    compute_size_distance_features,
    edge_min_scale,
    merge_bridge_floor,
    should_reject_field_scale,
)


def _feat_far_small(img: int = 256) -> object:
    # ~8x8 on 256 → small area; prox=0 → far=1
    return compute_size_distance_features(
        w=8,
        h=8,
        img_hw=(img, img),
        proximity_crop=np.zeros((8, 8), np.float32),
    )


def _feat_near_large(img: int = 200) -> object:
    return compute_size_distance_features(
        w=120,
        h=120,
        img_hw=(img, img),
        proximity_crop=np.ones((120, 120), np.float32),
    )


def _annotate(det: dict, feat) -> dict:
    det = dict(det)
    det["relative_far"] = feat.relative_far
    det["apparent_size"] = feat.apparent_size
    det["area_ratio"] = feat.area_ratio
    det["metric_proxy"] = feat.metric_proxy
    det["metric_size"] = feat.metric_size
    det["size_distance_band"] = feat.band
    return det


def test_lite_far_small_band_and_looser_gates():
    far_small = _feat_far_small()
    near_same_area = compute_size_distance_features(
        w=8,
        h=8,
        img_hw=(256, 256),
        proximity_crop=np.ones((8, 8), np.float32),
    )
    assert far_small.band == "far_small"
    assert far_small.apparent_size < near_same_area.apparent_size
    assert area_min_scale(far_small) < area_min_scale(_feat_near_large())
    assert edge_min_scale(far_small) < edge_min_scale(_feat_near_large())


def test_lite_far_small_weak_support_kept():
    """Uzak-küçük: zayıf support + ince sinyal → keep soft."""
    feat = _feat_far_small()
    det = _annotate(
        {
            "x": 100,
            "y": 100,
            "w": 8,
            "h": 8,
            "combined_pool": 0.005,
            "detector_conf": 0.004,
            "comb_mean": 0.02,
            "edge_mean": 0.015,
            "score": 0.01,
            "proposal_source": "heuristic",
        },
        feat,
    )
    assert feat.band == "far_small"
    assert _should_keep_detection(det, (256, 256, 3)) is True


def test_lite_near_large_pair_rejects_weak_bridge_merge():
    """Yakın-büyük ×2: span mean ~0.06 → legacy merge eder, floor engeller."""
    feat = _feat_near_large(200)
    assert feat.band == "near_large"
    assert merge_bridge_floor(feat, feat, 0.05) >= 0.12

    combined = np.full((200, 200), 0.06, dtype=np.float32)
    combined[40:90, 115:125] = 0.09
    a = _annotate({"x": 10, "y": 30, "w": 100, "h": 80, "score": 0.3}, feat)
    b = _annotate({"x": 130, "y": 35, "w": 60, "h": 70, "score": 0.3}, feat)
    diag = float(np.hypot(200, 200))
    # Aynı geometri, band yok → mid/legacy hâlâ birleşir (kontrast)
    assert _should_merge_proposals(
        {"x": 10, "y": 30, "w": 100, "h": 80},
        {"x": 130, "y": 35, "w": 60, "h": 70},
        combined,
        diag,
        0.15,
        0.5,
    ) is True
    assert _should_merge_proposals(a, b, combined, diag, merge_iou=0.15, merge_tol=0.5) is False


def test_lite_mid_fragments_still_merge_on_strong_bridge():
    """Mid/legacy: güçlü köprü + hizalı fragmanlar hâlâ birleşebilir."""
    combined = np.zeros((64, 64), dtype=np.float32)
    combined[20:30, 10:20] = 0.30
    combined[20:30, 20:34] = 0.08
    combined[20:30, 34:44] = 0.32
    a = {"x": 10, "y": 20, "w": 10, "h": 10}
    b = {"x": 34, "y": 20, "w": 10, "h": 10}
    assert _should_merge_proposals(a, b, combined, diag=float(np.hypot(64, 64)), merge_iou=0.15, merge_tol=0.5) is True


def test_lite_field_scale_near_large_rejected():
    feat = _feat_near_large(200)
    assert should_reject_field_scale(feat, support=0.02) is True
    det = _annotate(
        {
            "x": 10,
            "y": 10,
            "w": 120,
            "h": 120,
            "combined_pool": 0.02,
            "detector_conf": 0.015,
            "score": 0.02,
            "proposal_source": "heuristic",
        },
        feat,
    )
    assert _should_keep_detection(det, (200, 200, 3)) is False


def test_lite_same_area_near_vs_far_apparent():
    near = compute_size_distance_features(
        w=16,
        h=16,
        img_hw=(128, 128),
        proximity_crop=np.full((16, 16), 0.95, np.float32),
    )
    far = compute_size_distance_features(
        w=16,
        h=16,
        img_hw=(128, 128),
        proximity_crop=np.full((16, 16), 0.05, np.float32),
    )
    assert far.relative_far > near.relative_far
    assert far.apparent_size < near.apparent_size
