"""PR metric tests: AP, trapezoidal, tie-order invariance."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from detection_metrics_lib import (  # noqa: E402
    average_precision,
    average_precision_sklearn_ref,
    trapezoidal_pr_auc,
)


def test_ap_and_trap_bounds():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    ap = average_precision(y, s)
    tr = trapezoidal_pr_auc(y, s)
    assert ap is not None and 0 <= ap <= 1
    assert tr is not None and 0 <= tr <= 1


def test_auprc_tie_order_invariance():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    ap1 = average_precision(y, s)
    # permute
    order = np.array([3, 1, 0, 2])
    ap2 = average_precision(y[order], s[order])
    assert ap1 == ap2


def test_ap_matches_sklearn_when_no_ties():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.55, 0.7, 0.3, 0.9])
    ours = average_precision(y, s)
    ref = average_precision_sklearn_ref(y, s)
    if ref is None:
        pytest.skip("sklearn unavailable")
    assert ours is not None
    assert abs(ours - ref) < 1e-9
