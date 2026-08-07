"""Smoke test for keep/drop reason instrumentation (no torch)."""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_should_keep_detection_returns_tuple_in_source():
    """Guard: _should_keep_detection must return (bool, reason) without importing torch stack."""
    src = (REPO / "src/artps_detection_core.py").read_text(encoding="utf-8")
    assert "def _should_keep_detection" in src
    assert 'return False, "field_scale_rejection"' in src
    assert 'return True, "kept"' in src


def test_predict_image_accepts_diagnostics_out_param():
    src = (REPO / "src/artps_inference.py").read_text(encoding="utf-8")
    assert "diagnostics_out: MutableMapping[str, Any] | None = None" in src
    assert "diagnostics_candidates=" in src


def test_component_runner_refuses_test_split_in_source():
    src = (REPO / "scripts/iac2026/run_component_diagnostics_validation.py").read_text(
        encoding="utf-8"
    )
    assert 'refusing non-validation split' in src
    tree = ast.parse(src)
    assert tree is not None
