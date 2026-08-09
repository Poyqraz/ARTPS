"""Visualization-only candidate-support overlay (no scoring)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

cv2 = pytest.importorskip("cv2")

from _candidate_support_overlay import (  # noqa: E402
    FOOTPRINT_ALPHA,
    OVERLAY_VISUALIZATION_VERSION,
    draw_candidate_support_overlay,
    overlay_geometry_counts,
)


def _greenish(px: np.ndarray) -> bool:
    return int(px[1]) > 120 and int(px[1]) > int(px[0]) + 40 and int(px[1]) > int(px[2]) + 20


def test_overlay_version_is_v2():
    assert OVERLAY_VISUALIZATION_VERSION == "candidate_support_v2"
    assert FOOTPRINT_ALPHA <= 0.20


def test_open_corners_not_full_rectangle():
    rgb = np.zeros((80, 100, 3), np.uint8)
    rgb[:] = (40, 30, 20)
    combined = np.zeros((80, 100), np.float32)
    combined[35, 45] = 1.0
    dets = [{"x": 20, "y": 20, "w": 40, "h": 30, "score": 0.2, "support_geometry": "none"}]
    out = draw_candidate_support_overlay(rgb, dets, combined)
    mid_top = out[20, 40]
    corner = out[20, 20]
    assert _greenish(corner), corner.tolist()
    assert not _greenish(mid_top), mid_top.tolist()


def test_cc_footprint_tints_interior_not_bright_edge():
    rgb = np.zeros((60, 60, 3), np.uint8)
    rgb[:] = (30, 30, 30)
    combined = np.zeros((60, 60), np.float32)
    combined[28, 32] = 0.9
    contour = [[18, 18], [42, 18], [42, 42], [18, 42]]
    dets = [
        {
            "x": 15,
            "y": 15,
            "w": 30,
            "h": 30,
            "score": 0.1,
            "support_contour": contour,
            "support_geometry": "cc",
        }
    ]
    out = draw_candidate_support_overlay(rgb, dets, combined)
    interior = out[22, 22]
    outside = out[8, 8]
    assert int(interior[1]) > int(outside[1]) or int(interior[2]) > int(outside[2]), interior.tolist()
    assert int(interior[2]) < 180, interior.tolist()
    edge = out[18, 30]
    assert not (int(edge[2]) > 180 and int(edge[2]) > int(edge[1]) + 40), edge.tolist()
    ay, ax = 28, 32
    patch = out[max(0, ay - 2) : ay + 3, max(0, ax - 2) : ax + 3]
    assert (patch.max(axis=2) > 180).any()
    counts = overlay_geometry_counts(dets)
    assert counts == {"n_support_contour": 1, "n_oriented_poly": 0, "n_bracket_fallback": 0}


def test_peak_fallback_no_invented_footprint():
    rgb = np.zeros((50, 60, 3), np.uint8)
    rgb[:] = (20, 20, 20)
    combined = np.ones((50, 60), np.float32) * 0.1
    combined[22, 24] = 0.8
    dets = [
        {
            "x": 8,
            "y": 10,
            "w": 40,
            "h": 22,
            "score": 0.05,
            "support_geometry": "none",
            "peak_xy": [24, 22],
        }
    ]
    out = draw_candidate_support_overlay(rgb, dets, combined)
    counts = overlay_geometry_counts(dets)
    assert counts["n_bracket_fallback"] == 1
    assert counts["n_support_contour"] == 0
    assert not _greenish(out[10, 28])
    interior = out[16, 35]
    assert abs(int(interior[0]) - 20) < 8
    assert abs(int(interior[1]) - 20) < 8
    assert abs(int(interior[2]) - 20) < 8
    patch = out[20:25, 22:27]
    assert (patch.max(axis=2) > 180).any()


def test_labels_only_when_few_candidates():
    rgb = np.zeros((40, 80, 3), np.uint8)
    combined = np.zeros((40, 80), np.float32)
    two = [
        {"x": 5, "y": 5, "w": 12, "h": 12, "score": 0.2, "support_geometry": "none"},
        {"x": 30, "y": 5, "w": 12, "h": 12, "score": 0.1, "support_geometry": "none"},
    ]
    out2 = draw_candidate_support_overlay(rgb, two, combined)
    assert out2.sum() > rgb.sum()
    many = [
        {"x": i * 8, "y": 4, "w": 6, "h": 6, "score": 0.1, "support_geometry": "none"}
        for i in range(5)
    ]
    counts = overlay_geometry_counts(many)
    assert counts["n_bracket_fallback"] == 5
