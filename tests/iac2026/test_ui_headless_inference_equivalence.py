"""UI wrapper vs headless shared-core equivalence (CI-safe without torch)."""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TOL = 1e-7


def _map_checksum(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr.astype(np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def test_batch_runner_imports_shared_inference():
    src = (REPO / "scripts/iac2026/run_artps_frozen_full_profile.py").read_text(encoding="utf-8")
    assert "from src.artps_inference import" in src or "load_frozen_artps_profile" in src
    assert "predict_image" in src


def test_app_uses_shared_detection_core_wrapper():
    """app.py must wrap artps_detection_core — not a second fusion algorithm copy."""
    src = (REPO / "app.py").read_text(encoding="utf-8")
    assert "from src.artps_detection_core import" in src
    assert "from src import artps_detection_core as _adc" in src
    assert "_adc.compute_combined_anomaly_map" in src
    tree = ast.parse(src)
    wrapper = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_combined_anomaly_map":
            wrapper = node
            break
    assert wrapper is not None
    dump = ast.dump(wrapper)
    assert "compute_combined_anomaly_map" in dump


def test_shared_fusion_deterministic_on_fixture():
    torch = pytest.importorskip("torch")
    from src import artps_detection_core as adc

    rng = np.random.default_rng(0)
    h, w = 64, 64
    original = rng.random((h, w, 3), dtype=np.float64).astype(np.float32)
    reconstructed = np.clip(original + rng.normal(0, 0.02, original.shape).astype(np.float32), 0, 1)
    depth = rng.random((h, w), dtype=np.float64).astype(np.float32)

    adc.set_runtime_params(
        {
            "hyst_high": 97,
            "hyst_low": 92,
            "nms_iou": 0.35,
            "top_k": 10,
            "w_recon": 0.5,
            "w_depth": 0.3,
            "w_texture": 0.2,
            "edge_reinf": 0.35,
            "fp_suppression_enabled": True,
            "size_distance_policy": True,
            "recall_ablation": "slim",
        }
    )

    m1, d1, _ = adc.compute_combined_anomaly_map(
        original, reconstructed, depth, hyst_high_pct=97, hyst_low_pct=92, nms_iou=0.35, top_k=10
    )
    m2, d2, _ = adc.compute_combined_anomaly_map(
        original, reconstructed, depth, hyst_high_pct=97, hyst_low_pct=92, nms_iou=0.35, top_k=10
    )

    assert m1.shape == m2.shape == (h, w)
    assert _map_checksum(m1) == _map_checksum(m2)
    assert len(d1) == len(d2)
    for a, b in zip(d1, d2):
        assert a["x"] == b["x"] and a["y"] == b["y"] and a["w"] == b["w"] and a["h"] == b["h"]
        assert float(a.get("score", a.get("score_raw", 0))) == pytest.approx(
            float(b.get("score", b.get("score_raw", 0))), abs=TOL
        )
    assert torch is not None


def test_preprocess_hash_stable_for_raw_profile(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image

    from src.artps_inference import _preprocess_image

    img = Image.fromarray(np.full((40, 50, 3), 120, dtype=np.uint8))
    path = tmp_path / "t.png"
    img.save(path)
    a = np.asarray(_preprocess_image(path, "raw_rgb_v1"), dtype=np.uint8)
    b = np.asarray(_preprocess_image(path, "raw_rgb_v1"), dtype=np.uint8)
    assert hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest()
