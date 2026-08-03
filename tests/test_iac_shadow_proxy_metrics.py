"""Unit tests for IAC shadow proxy metrics."""
from __future__ import annotations

import numpy as np

from src.eval.iac_shadow_proxy import (
    aggregate_shadow_summaries,
    count_matched,
    is_flat_shadow_fp,
    is_rover_fp,
    is_shadow_dense,
    is_shadow_rock,
    summarize_image_pair,
)


def test_shadow_dense_and_flat_fp():
    shadow = np.full((64, 64), 0.4, np.float32)
    gate = np.zeros((64, 64), np.float32)
    assert is_shadow_dense(shadow) is True
    det = {"x": 10, "y": 10, "w": 20, "h": 20}
    assert is_flat_shadow_fp(det, shadow, gate) is True
    gate[10:30, 10:30] = 0.8
    assert is_flat_shadow_fp(det, shadow, gate) is False


def test_rover_and_shadow_rock():
    rover = np.zeros((64, 64), np.float32)
    rover[0:40, 40:64] = 1.0
    det = {"x": 42, "y": 5, "w": 15, "h": 20}
    assert is_rover_fp(det, rover) is True
    gate = np.zeros((64, 64), np.float32)
    gate[5:25, 42:57] = 0.5
    assert is_shadow_rock(det, gate) is True


def test_count_matched_and_pair_summary():
    off = [{"x": 10, "y": 10, "w": 20, "h": 20, "score": 0.05}]
    on = [{"x": 12, "y": 11, "w": 18, "h": 18, "score": 0.04}]
    assert count_matched(off, on) == 1
    shadow = np.full((64, 64), 0.3, np.float32)
    gate = np.full((64, 64), 0.5, np.float32)
    rover = np.zeros((64, 64), np.float32)
    row = summarize_image_pair(
        class_label="rocky",
        dets_off=off,
        dets_on=on,
        shadow_like=shadow,
        object_gate=gate,
        rover_body=rover,
    )
    assert row["shadow_dense"] is True
    assert row["recall_proxy"] == 1.0
    assert row["shadow_rock_loss"] == 0.0
    agg = aggregate_shadow_summaries([row])
    assert agg["n_shadow_dense"] == 1
    assert "disclaimer" in agg


def test_shadow_rock_loss_when_missing_on():
    off = [{"x": 10, "y": 10, "w": 20, "h": 20, "score": 0.05}]
    on: list[dict] = []
    gate = np.full((64, 64), 0.5, np.float32)
    row = summarize_image_pair(
        class_label="boulder",
        dets_off=off,
        dets_on=on,
        shadow_like=np.zeros((64, 64), np.float32),
        object_gate=gate,
        rover_body=np.zeros((64, 64), np.float32),
    )
    assert row["shadow_rock_loss"] == 1.0
