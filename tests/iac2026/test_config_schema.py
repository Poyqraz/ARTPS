"""Config schema validation tests."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from _config import (  # noqa: E402
    ConfigValidationError,
    apply_evaluation_purpose_policy,
    load_and_validate_config,
    validate_instance,
)


def test_synthetic_config_loads():
    cfg = load_and_validate_config(
        REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml"
    )
    assert cfg.evidence_mode == "software_verification"
    assert cfg["evaluation_purpose"] == "software_verification"


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


def test_independent_example_schema_ok_and_real_evidence_blocked():
    path = REPO / "reproduction/iac2026/configs/independent_evaluation.example.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["evaluation_purpose"] == "current_reproducible_evaluation"
    assert raw["claim_ids"] == ["IND_EVAL_V1"]
    errs = validate_instance(raw, "detection_reproduction_config.schema.json")
    assert errs == []
    # Pending checkpoint/data: load must fail closed on real_evidence policy
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(path)


def test_independent_purpose_rejects_c05_claim_ids():
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/independent_evaluation.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["claim_ids"] = ["C05"]
    schema_errs = validate_instance(raw, "detection_reproduction_config.schema.json")
    assert schema_errs, "schema must reject C05 under current_reproducible_evaluation"
    policy_errs = apply_evaluation_purpose_policy(raw)
    assert any("IND_EVAL_V1" in e or "claim_ids" in e for e in policy_errs)


def test_independent_sw_config_loads_when_purpose_ok(tmp_path):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw = copy.deepcopy(raw)
    raw["evaluation_purpose"] = "current_reproducible_evaluation"
    raw["claim_ids"] = ["IND_EVAL_V1"]
    raw["task_level"] = "image_binary"
    raw["threshold_policy"] = "validation_selected"
    raw["threshold_selection_metric"] = "f1"
    raw["threshold_tie_break"] = "highest_threshold"
    raw["fixed_threshold"] = None
    raw["pr_metric_method"] = "average_precision"
    p = tmp_path / "indep_sw.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_and_validate_config(p)
    assert cfg["claim_ids"] == ["IND_EVAL_V1"]
    assert cfg["evaluation_purpose"] == "current_reproducible_evaluation"
