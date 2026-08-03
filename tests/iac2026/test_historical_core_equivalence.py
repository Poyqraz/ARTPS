"""Historical core golden equivalence (software verification)."""
from __future__ import annotations

import hashlib
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
    # Stable map fingerprint for regression
    digest = hashlib.sha256(np.ascontiguousarray(combined_a).tobytes()).hexdigest()
    fixture = REPO / "reproduction/iac2026/fixtures/historical_core_golden.json"
    payload = {
        "map_sha256": digest,
        "n_detections": len(dets_a),
        "map_mean": float(np.mean(combined_a)),
        "map_max": float(np.max(combined_a)),
    }
    if not fixture.exists():
        fixture.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    assert digest == expected["map_sha256"]
    assert len(dets_a) == expected["n_detections"]
    assert abs(payload["map_mean"] - expected["map_mean"]) < 1e-5
    assert "total_pipeline" in stages
