from app import _bridge_strength, _collect_peak_window_detections, _should_merge_proposals

import numpy as np


def test_bridge_strength_is_high_for_connected_regions():
    combined = np.zeros((64, 64), dtype=np.float32)
    combined[20:30, 10:20] = 0.30
    combined[20:30, 20:34] = 0.08
    combined[20:30, 34:44] = 0.32
    a = {"x": 10, "y": 20, "w": 10, "h": 10}
    b = {"x": 34, "y": 20, "w": 10, "h": 10}
    assert _bridge_strength(combined, a, b) > 0.05


def test_should_merge_proposals_for_aligned_connected_fragments():
    combined = np.zeros((64, 64), dtype=np.float32)
    combined[20:30, 10:20] = 0.30
    combined[20:30, 20:34] = 0.08
    combined[20:30, 34:44] = 0.32
    a = {"x": 10, "y": 20, "w": 10, "h": 10}
    b = {"x": 34, "y": 20, "w": 10, "h": 10}
    assert _should_merge_proposals(a, b, combined, diag=float(np.hypot(64, 64)), merge_iou=0.15, merge_tol=0.5) is True


def test_should_merge_proposals_rejects_distant_weak_bridge():
    """Rocky field: zayıf köprü + uzak merkez → birleşme yok."""
    combined = np.full((128, 128), 0.03, dtype=np.float32)
    combined[20:40, 10:30] = 0.25
    combined[20:40, 80:100] = 0.25
    a = {"x": 10, "y": 20, "w": 20, "h": 20}
    b = {"x": 80, "y": 20, "w": 20, "h": 20}
    assert _should_merge_proposals(a, b, combined, diag=float(np.hypot(128, 128)), merge_iou=0.15, merge_tol=0.5) is False


def test_collect_peak_window_detections_recovers_local_peaks():
    seed = np.zeros((64, 64), dtype=np.float32)
    seed[18:24, 20:26] = 0.12
    seed[40:46, 44:50] = 0.14
    dets = _collect_peak_window_detections(seed, top_k=4, rover_body=None, boundary_shadow=None)
    assert len(dets) >= 2
    assert any(det["proposal_source"] == "heuristic_peaks" for det in dets)
