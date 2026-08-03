"""Size/distance 2x2 semantics unit tests."""
from __future__ import annotations

import numpy as np

from src.core.size_distance import (
    area_min_scale,
    compute_size_distance_features,
    estimate_depth_scale_m,
    merge_bridge_floor,
    shadow_cut_delta,
    should_reject_field_scale,
)


def test_same_area_far_reduces_apparent_size() -> None:
    near = compute_size_distance_features(
        w=20, h=20, img_hw=(200, 200),
        proximity_crop=np.full((20, 20), 0.9, np.float32),
    )
    far = compute_size_distance_features(
        w=20, h=20, img_hw=(200, 200),
        proximity_crop=np.full((20, 20), 0.1, np.float32),
    )
    assert far.relative_far > near.relative_far
    assert far.apparent_size < near.apparent_size


def test_band_labels() -> None:
    far_small = compute_size_distance_features(
        w=8, h=8, img_hw=(256, 256),
        proximity_crop=np.zeros((8, 8), np.float32),
    )
    assert far_small.band == "far_small"
    near_large = compute_size_distance_features(
        w=120, h=120, img_hw=(200, 200),
        proximity_crop=np.ones((120, 120), np.float32),
    )
    assert near_large.band == "near_large"


def test_area_min_scale_far_small_looser_than_near_large() -> None:
    fs = compute_size_distance_features(
        w=8, h=8, img_hw=(256, 256),
        proximity_crop=np.zeros((8, 8), np.float32),
    )
    nl = compute_size_distance_features(
        w=120, h=120, img_hw=(200, 200),
        proximity_crop=np.ones((120, 120), np.float32),
    )
    assert area_min_scale(fs) < area_min_scale(nl)


def test_merge_bridge_floor_near_large_pair() -> None:
    nl = compute_size_distance_features(
        w=120, h=120, img_hw=(200, 200),
        proximity_crop=np.ones((120, 120), np.float32),
    )
    assert merge_bridge_floor(nl, nl, 0.05) > 0.05


def test_depth_none_neutral() -> None:
    feat = compute_size_distance_features(w=16, h=16, img_hw=(128, 128), depth_crop=None)
    assert abs(feat.relative_far - 0.5) < 1e-6
    assert feat.band in {"mid", "near_small", "far_small", "near_large", "far_large"}


def test_metric_proxy_and_scale() -> None:
    bare = compute_size_distance_features(
        w=16, h=16, img_hw=(128, 128),
        proximity_crop=np.full((16, 16), 0.5, np.float32),
    )
    assert bare.metric_proxy is None
    assert bare.metric_size is None
    with_span = compute_size_distance_features(
        w=16, h=16, img_hw=(128, 128),
        proximity_crop=np.full((16, 16), 0.5, np.float32),
        depth_span=0.8,
    )
    assert with_span.metric_proxy is not None
    scaled = compute_size_distance_features(
        w=16, h=16, img_hw=(128, 128),
        proximity_crop=np.full((16, 16), 0.5, np.float32),
        depth_span=0.8,
        depth_scale_m=2.0,
    )
    assert scaled.metric_size is not None
    assert estimate_depth_scale_m(np.ones((8, 8))) is None


def test_shadow_cut_delta_and_field_reject() -> None:
    far = compute_size_distance_features(
        w=8, h=8, img_hw=(256, 256),
        proximity_crop=np.zeros((8, 8), np.float32),
    )
    assert shadow_cut_delta(far) > 0.0
    big = compute_size_distance_features(
        w=150, h=150, img_hw=(200, 200),
        proximity_crop=np.ones((150, 150), np.float32),
        depth_span=0.2,
    )
    assert big.band == "near_large"
    assert should_reject_field_scale(big, support=0.01) is True
    assert should_reject_field_scale(big, support=0.5) is False
    mid = compute_size_distance_features(
        w=40, h=30, img_hw=(128, 128),
    )
    assert should_reject_field_scale(mid, support=0.012) is False
