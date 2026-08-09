"""Visualization-only support geometry must not change xywh or scores."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

cv2 = pytest.importorskip("cv2")

from src.artps_detection_core import (  # noqa: E402
    _copy_viz_support,
    _union_support_contours,
    compute_combined_anomaly_map,
)

FIXTURE = REPO / "tests" / "iac2026" / "fixtures" / "qualitative_overlay_candidates.json"


def test_copy_viz_support_does_not_touch_score_xywh():
    src = {
        "x": 1,
        "y": 2,
        "w": 10,
        "h": 8,
        "score": 0.42,
        "poly": None,
        "support_contour": [[1, 2], [10, 2], [10, 9], [1, 9]],
        "support_geometry": "cc",
        "peak_xy": [4, 5],
    }
    dst = {"x": 1, "y": 2, "w": 10, "h": 8, "score": 0.42, "poly": None, "proposal_source": "heuristic"}
    _copy_viz_support(src, dst)
    assert dst["x"] == 1 and dst["y"] == 2 and dst["w"] == 10 and dst["h"] == 8
    assert dst["score"] == 0.42
    assert dst["poly"] is None
    assert dst["support_geometry"] == "cc"
    assert dst["support_contour"] == src["support_contour"]
    assert dst["peak_xy"] == [4, 5]


def test_union_support_contours_uses_existing_ccs_only():
    a = {"support_contour": [[5, 5], [15, 5], [15, 15], [5, 15]]}
    b = {"support_contour": [[12, 12], [22, 12], [22, 22], [12, 22]]}
    union = _union_support_contours([a, b], (40, 40))
    assert union is not None and len(union) >= 4
    assert _union_support_contours([a, {"x": 0}], (40, 40)) is None


def test_combined_map_heuristic_dets_keep_cc_without_score_shape_change():
    rgb = np.full((64, 64, 3), 0.45, np.float32)
    rgb[20:40, 20:40] = 0.85
    recon = np.full((64, 64, 3), 0.45, np.float32)
    depth = np.linspace(0.1, 0.9, 64, dtype=np.float32)[:, None].repeat(64, axis=1)
    combined, dets, _diag = compute_combined_anomaly_map(
        rgb,
        recon,
        depth,
        hyst_high_pct=96,
        hyst_low_pct=90,
        top_k=5,
    )
    assert combined.shape == (64, 64)
    for det in dets:
        assert {"x", "y", "w", "h", "score"} <= set(det)
        assert det.get("support_geometry") in {"cc", "oriented_poly", "none"}
        if det.get("support_geometry") == "cc":
            cnt = det.get("support_contour")
            assert isinstance(cnt, list) and len(cnt) >= 3


def test_overlay_fixture_file_exists_and_has_locked_counts():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["fig2"]["n_valid_candidates"] == 2
    assert data["fig2"]["n_raw_detections"] == 3
    assert data["fig4_close"]["n_valid_candidates"] == 6
    assert data["fig4_far"]["n_valid_candidates"] == 5
    assert len(data["fig2"]["candidates"]) == 2
    assert data["fig3_png_sha256"]
