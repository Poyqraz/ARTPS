"""Analytical PaDiM Mahalanobis einsum contract (numpy-only for CI).

Canonical implementation: src/models/anomaly/padim.py
Legacy mirror: ARTPS/src/models/anomaly/padim.py (not refactored in this PR).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
EINSUM = 'torch.einsum("ni,ij,nj->n"'


def _mahalanobis_sq(X: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    # Same contraction as padim: einsum("ni,ij,nj->n", X, cov_inv, X)
    return np.einsum("ni,ij,nj->n", X, cov_inv, X)


def test_identity_covariance_is_euclidean():
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    cov_inv = np.eye(2, dtype=np.float64)
    d2 = _mahalanobis_sq(X, cov_inv)
    assert np.isfinite(d2).all()
    assert float(d2[0]) == pytest.approx(25.0)
    assert float(d2[1]) == pytest.approx(0.0)
    assert float(d2[2]) == pytest.approx(1.0)
    assert (d2 >= 0).all()
    dist = np.sqrt(d2)
    assert float(dist[0]) == pytest.approx(5.0)


def test_known_diagonal_metric():
    X = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    cov_inv = np.diag(np.array([4.0, 1.0], dtype=np.float64))
    d2 = _mahalanobis_sq(X, cov_inv)
    assert float(d2[0]) == pytest.approx(16.0)
    assert float(d2[1]) == pytest.approx(9.0)


def test_spatial_reshape_finite():
    H, W, C = 4, 4, 3
    rng = np.random.default_rng(0)
    X = rng.standard_normal((H * W, C)).astype(np.float64)
    cov = np.eye(C, dtype=np.float64)
    d2 = _mahalanobis_sq(X, cov)
    assert d2.shape == (H * W,)
    amap = np.sqrt(np.maximum(d2, 0.0)).reshape(H, W)
    assert amap.shape == (H, W)
    assert np.isfinite(amap).all()
    assert (d2 >= -1e-12).all()


def test_asymmetric_cov_inv_symmetrize_or_finite():
    A = np.array([[2.0, 0.3], [0.1, 1.5]], dtype=np.float64)
    cov_inv = 0.5 * (A + A.T)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 2)).astype(np.float64)
    d2 = _mahalanobis_sq(X, cov_inv)
    assert np.isfinite(d2).all()
    assert (d2 >= -1e-9).all()


def test_src_and_artps_padim_share_einsum():
    src = (REPO / "src/models/anomaly/padim.py").read_text(encoding="utf-8")
    artps = (REPO / "ARTPS/src/models/anomaly/padim.py").read_text(encoding="utf-8")
    assert EINSUM in src
    assert EINSUM in artps
    assert 'einsum("nc,cc,nd->n"' not in src
    assert 'einsum("nc,cc,nd->n"' not in artps
