"""Config schema validation tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from _config import ConfigValidationError, load_and_validate_config  # noqa: E402


def test_synthetic_config_loads():
    cfg = load_and_validate_config(
        REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml"
    )
    assert cfg.evidence_mode == "software_verification"


def test_example_real_evidence_rejected_for_tbd(tmp_path):
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(
            REPO / "reproduction/iac2026/configs/detection_reproduction.example.yaml"
        )


def test_invalid_task_level(tmp_path):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["task_level"] = "not_a_level"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(p)


def test_missing_config_field(tmp_path):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    del raw["pr_metric_method"]
    p = tmp_path / "miss.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(p)


def test_bootstrap_nonzero_rejected(tmp_path):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["bootstrap_iterations"] = 100
    p = tmp_path / "boot.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="bootstrap"):
        load_and_validate_config(p)
