"""Layer C soft-similarity penalty: s(r) in [0,1] so S' never exceeds S."""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb + 1e-8))


def _s_r(z: np.ndarray, hist: list) -> float:
    if not hist:
        return 0.0
    return max(0.0, max(_cosine_sim(z, h) for h in hist))


def _s_prime(raw: float, sim_lambda: float, s: float) -> float:
    return float(max(0.0, raw * (1.0 - sim_lambda * s)))


def test_app_clips_layer_c_cosine_at_penalty_site():
    src = (REPO / "app.py").read_text(encoding="utf-8")
    assert "sim_max = max(0.0, max(_cosine_sim(z, h) for h in hist))" in src


def test_negative_cosine_clamps_s_to_zero():
    z = np.array([1.0, 0.0], dtype=np.float32)
    hist = [np.array([-1.0, 0.0], dtype=np.float32)]
    assert _cosine_sim(z, hist[0]) < 0.0
    s = _s_r(z, hist)
    assert s == 0.0
    assert _s_prime(0.80, 0.50, s) == 0.80


def test_zero_similarity_applies_no_penalty():
    z = np.array([1.0, 0.0], dtype=np.float32)
    hist = [np.array([0.0, 1.0], dtype=np.float32)]
    s = _s_r(z, hist)
    assert s == 0.0
    assert _s_prime(0.80, 0.50, s) == 0.80


def test_positive_similarity_decreases_priority():
    z = np.array([1.0, 0.0], dtype=np.float32)
    hist = [np.array([1.0, 0.0], dtype=np.float32)]
    s = _s_r(z, hist)
    raw = 0.80
    primed = _s_prime(raw, 0.50, s)
    assert s > 0.0
    assert primed < raw


def test_s_prime_never_exceeds_s():
    raw = 0.80
    lam = 0.50
    cases = [
        (np.array([1.0, 0.0], dtype=np.float32), [np.array([-1.0, 0.0], dtype=np.float32)]),
        (np.array([1.0, 0.0], dtype=np.float32), [np.array([0.0, 1.0], dtype=np.float32)]),
        (np.array([1.0, 0.0], dtype=np.float32), [np.array([1.0, 0.0], dtype=np.float32)]),
    ]
    for z, hist in cases:
        s = _s_r(z, hist)
        assert 0.0 <= s <= 1.0
        assert _s_prime(raw, lam, s) <= raw
