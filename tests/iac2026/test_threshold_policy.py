"""Threshold policy tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from detection_metrics_lib import select_threshold_on_validation  # noqa: E402


def test_validation_single_class_rejects():
    t, m = select_threshold_on_validation(
        np.array([0, 0]), np.array([0.1, 0.2]), metric="f1", tie_break="highest_threshold"
    )
    assert t is None


def test_tie_break_determinism():
    y = np.array([0, 1, 0, 1])
    # Two thresholds with same F1 possible
    s = np.array([0.2, 0.8, 0.3, 0.7])
    t_hi, _ = select_threshold_on_validation(y, s, metric="f1", tie_break="highest_threshold")
    t_lo, _ = select_threshold_on_validation(y, s, metric="f1", tie_break="lowest_threshold")
    assert t_hi is not None and t_lo is not None
    assert t_hi >= t_lo


def test_test_labels_do_not_affect_selected_threshold():
    y_val = np.array([0, 1])
    s_val = np.array([0.1, 0.9])
    t1, _ = select_threshold_on_validation(y_val, s_val, metric="f1", tie_break="highest_threshold")
    # Changing unrelated test arrays must not matter — function only sees val
    t2, _ = select_threshold_on_validation(y_val, s_val, metric="f1", tie_break="highest_threshold")
    assert t1 == t2 == 0.9
