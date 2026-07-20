"""Sentetik rocky/boulder proposal regression testleri."""
from app import (
    _boost_recall_detection_pools,
    _cap_detections_if_needed,
    _collect_detail_first_detections,
    _collect_plateau_detections,
    _fuse_with_plateau_detections,
    _is_clutter_mode,
    _is_rocky_recall_mode,
    _should_run_detail_first_recall,
    _collect_peak_window_detections,
    _should_keep_detection,
)

import numpy as np


def test_plateau_pass_merges_single_high_activation_blob():
    combined = np.zeros((128, 128), dtype=np.float32)
    combined[40:90, 30:95] = 0.55
    area_min = 0.001 * 128 * 128
    plateau = _collect_plateau_detections(combined, area_min=area_min, percentile=88.0)
    assert len(plateau) <= 2
    assert plateau[0]["w"] * plateau[0]["h"] > 0.15 * 128 * 128
    assert plateau[0]["proposal_source"] == "heuristic_plateau"
    assert float(plateau[0].get("fill_ratio", 0.0)) >= 0.40


def test_plateau_skips_sparse_rocky_field():
    """Dağınık küçük tepeler alan-ölçekli tek kutuya birleşmemeli."""
    combined = np.zeros((128, 128), dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(25):
        y = int(rng.integers(10, 110))
        x = int(rng.integers(10, 110))
        combined[y : y + 4, x : x + 4] = 0.6
    plateau = _collect_plateau_detections(combined, area_min=8.0, percentile=90.0)
    assert all(float(d.get("area_ratio", 1.0)) <= 0.12 or float(d.get("fill_ratio", 0.0)) >= 0.70 for d in plateau)
    assert all(float(d.get("fill_ratio", 0.0)) >= 0.40 for d in plateau)


def test_should_keep_rejects_oversized_merged_box():
    det = {
        "x": 40,
        "y": 80,
        "w": 400,
        "h": 280,
        "combined_pool": 0.04,
        "detector_conf": 0.02,
        "proposal_source": "heuristic_merged",
    }
    assert _should_keep_detection(det, (768, 768, 3)) is False


def test_fuse_plateau_replaces_fragment_boxes():
    combined = np.zeros((64, 64), dtype=np.float32)
    combined[20:40, 15:50] = 0.4
    frags = [
        {"x": 15, "y": 20, "w": 12, "h": 18, "score": 0.2, "proposal_source": "heuristic"},
        {"x": 28, "y": 22, "w": 10, "h": 16, "score": 0.18, "proposal_source": "heuristic"},
    ]
    plateau = _collect_plateau_detections(combined, area_min=50.0, percentile=85.0)
    fused = _fuse_with_plateau_detections(frags, plateau, iou_replace=0.3)
    assert len(fused) <= 2
    assert any(d.get("proposal_source") == "heuristic_plateau" for d in fused)


def test_clutter_mode_detects_weak_combined_strong_fine_detail():
    combined = np.full((64, 64), 0.05, dtype=np.float32)
    fine = np.full((64, 64), 0.02, dtype=np.float32)
    fine[8:56, 8:56] = 0.12
    assert _is_clutter_mode(combined, fine) is True


def test_clutter_peak_pass_recovers_small_rocks():
    combined = np.full((128, 128), 0.04, dtype=np.float32)
    fine = np.zeros((128, 128), dtype=np.float32)
    fine[22:28, 30:36] = 0.22
    fine[70:76, 80:86] = 0.20
    seed = np.clip(0.35 * combined + 0.65 * fine, 0.0, 1.0)
    dets = _collect_peak_window_detections(
        seed,
        top_k=4,
        rover_body=None,
        boundary_shadow=None,
        peak_percentile=98.5,
        window_scale=0.05,
    )
    assert len(dets) >= 1
    assert dets[0]["proposal_source"] == "heuristic_peaks"


def test_flat_combined_not_clutter_mode():
    combined = np.random.default_rng(0).random((64, 64)).astype(np.float32) * 0.3 + 0.2
    fine = np.zeros((64, 64), dtype=np.float32)
    assert _is_clutter_mode(combined, fine) is False


def test_detail_first_recall_recovers_local_fine_detail_clusters():
    combined = np.full((128, 128), 0.03, dtype=np.float32)
    fine = np.zeros((128, 128), dtype=np.float32)
    fine[24:34, 28:40] = 0.25
    fine[72:80, 84:96] = 0.22
    seed = np.clip(0.25 * combined + 0.75 * fine, 0.0, 1.0)
    dets = _collect_detail_first_detections(
        seed,
        top_k=6,
        area_min=10.0,
        rover_body=None,
        boundary_shadow=None,
    )
    assert len(dets) >= 1
    assert any(det["proposal_source"] == "heuristic_detail_first" for det in dets)


def test_should_keep_detection_rejects_wide_upper_band_merged_box():
    det = {
        "x": 0,
        "y": 0,
        "w": 768,
        "h": 144,
        "combined_pool": 0.025,
        "detector_conf": 0.012,
        "proposal_source": "heuristic_merged",
    }
    assert _should_keep_detection(det, (768, 768, 3)) is False


def test_should_run_detail_first_skips_when_enough_contour_proposals():
    combined = np.full((64, 64), 0.05, dtype=np.float32)
    fine = np.full((64, 64), 0.02, dtype=np.float32)
    fine[8:56, 8:56] = 0.12
    existing = [
        {"x": 10, "y": 10, "w": 20, "h": 20, "score": 0.2},
        {"x": 30, "y": 30, "w": 18, "h": 18, "score": 0.18},
    ]
    assert _should_run_detail_first_recall(combined, fine, existing) is False


def test_should_keep_detection_rejects_detail_first_bottom_horizon_band():
    det = {
        "x": 0,
        "y": 712,
        "w": 768,
        "h": 22,
        "combined_pool": 0.001,
        "comb_mean": 0.03,
        "edge_mean": 0.01,
        "detector_conf": 0.02,
        "proposal_source": "heuristic_detail_first",
    }
    assert _should_keep_detection(det, (768, 768, 3)) is False


def test_rocky_recall_mode_is_narrower_than_clutter_mode():
    combined = np.full((64, 64), 0.05, dtype=np.float32)
    fine = np.full((64, 64), 0.10, dtype=np.float32)
    assert _is_clutter_mode(combined, fine) is True
    assert _is_rocky_recall_mode(combined, fine) is False


def test_boost_recall_detection_pools_uses_seed_strength():
    det = {
        "proposal_source": "heuristic_detail_first",
        "combined_pool": 0.001,
        "comb_mean": 0.04,
        "edge_mean": 0.03,
        "score": 0.035,
        "detector_conf": 0.01,
    }
    _boost_recall_detection_pools(det)
    assert det["combined_pool"] >= 0.03
    assert det["detector_conf"] >= 0.04


def test_cap_detections_trims_fragment_spray_not_rocky_rich():
    combined = np.zeros((64, 64), dtype=np.float32)
    combined[20:40, 20:40] = 0.2
    fine = np.zeros((64, 64), dtype=np.float32)
    dets = [
        {"x": i, "y": 1, "w": 4, "h": 4, "score": 0.05 - i * 0.001, "proposal_source": "heuristic"}
        for i in range(8)
    ]
    capped = _cap_detections_if_needed(dets, combined, fine, max_default=4)
    assert len(capped) == 4


def test_should_keep_detection_keeps_weak_plateau_for_recall():
    det = {
        "x": 10,
        "y": 10,
        "w": 40,
        "h": 30,
        "combined_pool": 0.002,
        "score": 0.012,
        "plateau_mass": 12.0,
        "proposal_source": "heuristic_plateau",
    }
    assert _should_keep_detection(det, (128, 128, 3)) is True
