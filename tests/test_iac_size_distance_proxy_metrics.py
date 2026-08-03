"""Unit tests for IAC size/distance proxy metrics."""
from __future__ import annotations

from src.eval.iac_size_distance_proxy import (
    aggregate_size_distance_summaries,
    is_far_small_proxy,
    is_field_scale_fp,
    is_near_large_merged,
    mean_matched_iou,
    summarize_size_distance_image,
)


def test_far_small_and_field_scale_flags():
    fs = {"size_distance_band": "far_small", "area_ratio": 0.005, "relative_far": 0.8}
    assert is_far_small_proxy(fs) is True
    big = {"area_ratio": 0.15, "proposal_source": "heuristic"}
    assert is_field_scale_fp(big) is True
    merged = {
        "proposal_source": "heuristic_merged",
        "size_distance_band": "near_large",
        "area_ratio": 0.12,
    }
    assert is_near_large_merged(merged) is True


def test_mean_matched_iou_and_summary():
    off = [
        {
            "x": 10, "y": 10, "w": 8, "h": 8, "score": 0.05,
            "size_distance_band": "far_small", "area_ratio": 0.005, "relative_far": 0.7,
            "proposal_source": "heuristic",
        },
        {
            "x": 50, "y": 50, "w": 80, "h": 80, "score": 0.1,
            "size_distance_band": "near_large", "area_ratio": 0.15,
            "proposal_source": "heuristic_merged",
        },
    ]
    on = [
        {
            "x": 11, "y": 10, "w": 8, "h": 8, "score": 0.05,
            "size_distance_band": "far_small", "area_ratio": 0.005, "relative_far": 0.7,
            "proposal_source": "heuristic",
        },
    ]
    assert mean_matched_iou(off, on) is not None
    row = summarize_size_distance_image(class_label="rocky", dets_off=off, dets_on=on)
    assert row["far_small_recall"] == 1.0
    assert row["near_large_over_merge_off"] > 0.0
    assert row["field_scale_fpr_off"] > 0.0
    assert row["field_scale_fpr_on"] == 0.0
    agg = aggregate_size_distance_summaries([row])
    assert "disclaimer" in agg
    assert agg["far_small_recall_mean"] == 1.0
