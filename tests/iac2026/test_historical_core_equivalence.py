"""Historical core golden equivalence (software verification)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from cv_core_pipeline import (  # noqa: E402
    PIPELINE_ID,
    SOURCE_COMMIT,
    core_process_rgb_u8,
    process_frame_historical,
)


def _fixed_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    rgb[40:80, 40:80] = 255
    rgb[120:150, 160:200] = 0
    return rgb


def test_pipeline_metadata():
    assert PIPELINE_ID == "historical_opencv_surrogate_8f7e3ff"
    assert SOURCE_COMMIT == "8f7e3ff"


def test_historical_golden_equivalence():
    rgb = _fixed_frame()
    combined_a, dets_a = core_process_rgb_u8(rgb)
    combined_b, dets_b, stages = process_frame_historical(rgb, target_res=256)
    assert combined_a.shape == (256, 256)
    np.testing.assert_allclose(combined_a, combined_b, rtol=0, atol=1e-6)
    assert len(dets_a) == len(dets_b)
    assert float(np.min(combined_a)) >= 0.0
    assert float(np.max(combined_a)) <= 1.0 + 1e-6
    fixture = REPO / "reproduction/iac2026/fixtures/historical_core_golden.json"
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    n = len(dets_a)
    mean = float(np.mean(combined_a))
    mx = float(np.max(combined_a))
    assert expected["n_detections_min"] <= n <= expected["n_detections_max"]
    assert expected["map_mean_min"] <= mean <= expected["map_mean_max"]
    assert mx >= expected["map_max_min"]
    assert "total_pipeline" in stages
    # Same process should be numerically stable on one host (OpenCV may nudge ulps)
    combined_c, dets_c = core_process_rgb_u8(rgb)
    np.testing.assert_allclose(combined_a, combined_c, rtol=0, atol=1e-5)
    assert len(dets_a) == len(dets_c)
