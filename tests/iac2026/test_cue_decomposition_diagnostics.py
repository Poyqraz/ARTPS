"""Frozen fusion diagnostics expose the three primary cue maps."""
from __future__ import annotations

import numpy as np

from src.artps_detection_core import compute_combined_anomaly_map


def test_combined_map_diagnostics_include_fusion_cues():
    rgb = np.full((64, 64, 3), 0.45, np.float32)
    rgb[20:40, 20:40] = 0.85
    recon = np.full((64, 64, 3), 0.45, np.float32)
    depth = np.linspace(0.1, 0.9, 64, dtype=np.float32)[:, None].repeat(64, axis=1)
    combined, _dets, diag = compute_combined_anomaly_map(
        rgb,
        recon,
        depth,
        hyst_high_pct=96,
        hyst_low_pct=90,
        top_k=5,
    )
    h, w = combined.shape[:2]
    for key in ("recon_diff_n", "depth_edge_n", "texture_term", "raw_combined_pre_mask"):
        assert key in diag
        arr = np.asarray(diag[key])
        assert arr.shape == (h, w)
        assert np.isfinite(arr).all()
    assert diag["w_recon"] == 0.50
    assert diag["w_depth"] == 0.30
    assert diag["w_texture"] == 0.20
