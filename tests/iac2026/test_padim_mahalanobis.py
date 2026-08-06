"""Analytical PaDiM Mahalanobis einsum contract.

Canonical implementation: src/models/anomaly/padim.py
Legacy mirror: ARTPS/src/models/anomaly/padim.py (not refactored in this PR).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
EINSUM = 'torch.einsum("ni,ij,nj->n"'


def _mahalanobis(X: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
    # Same formula as padim.predict_anomaly_map
    return torch.einsum("ni,ij,nj->n", X, cov_inv, X)


def test_identity_covariance_is_euclidean():
    X = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    cov_inv = torch.eye(2, dtype=torch.float64)
    d2 = _mahalanobis(X, cov_inv)
    assert torch.isfinite(d2).all()
    assert float(d2[0]) == pytest.approx(25.0)
    assert float(d2[1]) == pytest.approx(0.0)
    assert float(d2[2]) == pytest.approx(1.0)
    # distances before sqrt are non-negative
    assert (d2 >= 0).all()
    dist = d2.sqrt()
    assert float(dist[0]) == pytest.approx(5.0)


def test_known_diagonal_metric():
    X = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    cov_inv = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64))
    d2 = _mahalanobis(X, cov_inv)
    assert float(d2[0]) == pytest.approx(16.0)  # 2^2 * 4
    assert float(d2[1]) == pytest.approx(9.0)  # 3^2 * 1


def test_spatial_reshape_finite():
    H, W, C = 4, 4, 3
    X = torch.randn(H * W, C, dtype=torch.float64)
    cov = torch.eye(C, dtype=torch.float64)
    d2 = _mahalanobis(X, cov)
    assert d2.shape == (H * W,)
    amap = d2.sqrt().reshape(H, W)
    assert amap.shape == (H, W)
    assert torch.isfinite(amap).all()
    assert (d2 >= -1e-12).all()


def test_asymmetric_cov_inv_symmetrize_or_finite():
    # Fail-loud / numerical policy: symmetrized inverse keeps distances real.
    A = torch.tensor([[2.0, 0.3], [0.1, 1.5]], dtype=torch.float64)
    cov_inv = 0.5 * (A + A.T)
    X = torch.randn(8, 2, dtype=torch.float64)
    d2 = _mahalanobis(X, cov_inv)
    assert torch.isfinite(d2).all()
    assert (d2 >= -1e-9).all()


def test_src_and_artps_padim_share_einsum():
    src = (REPO / "src/models/anomaly/padim.py").read_text(encoding="utf-8")
    artps = (REPO / "ARTPS/src/models/anomaly/padim.py").read_text(encoding="utf-8")
    assert EINSUM in src
    assert EINSUM in artps
    # Old buggy form must not reappear
    assert 'einsum("nc,cc,nd->n"' not in src
    assert 'einsum("nc,cc,nd->n"' not in artps
