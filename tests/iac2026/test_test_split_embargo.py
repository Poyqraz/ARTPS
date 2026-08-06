"""Test-split embargo helper tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from test_split_embargo import assert_split_allowed, is_test_split_open, load_test_open_status  # noqa: E402


def test_checked_in_status_is_closed():
    status = load_test_open_status()
    assert status.get("test_opened") is False


def test_final_test_scope_pinned_closed():
    """Next PR must use frozen mars_clf_on / FP32 / threshold 0.0; test stays closed."""
    scope = yaml.safe_load(
        (REPO / "reproduction/iac2026/test_freeze/FINAL_TEST_SCOPE.yaml").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (REPO / "results/iac2026/independent_eval_v1/validation/profile_selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert scope["test_opened"] is False
    assert scope["selected_config_id"] == "artps_full_frozen_mars_clf_on_v1"
    assert scope["precision_mode"] == "fp32"
    assert scope["selected_threshold"] == 0.0
    assert scope["amp_primary_allowed"] is False
    assert scope["selected_config_id"] == selection["selected_config_id"]
    assert float(scope["selected_threshold"]) == float(selection["selected_threshold"])
    assert scope["validation_selection_artifact_sha256"] == selection["artifact_sha256"]
    assert "new_validation_profiles" in scope["forbidden_in_final_test_pr"]


def test_assert_split_allowed_validation_ok():
    assert_split_allowed("validation", {"test_opened": False})


def test_assert_split_allowed_test_refused():
    with pytest.raises(ValueError, match="test_opened=false"):
        assert_split_allowed("test", {"test_opened": False, "reason": "embargo"})


def test_assert_split_allowed_test_open():
    assert_split_allowed("test", {"test_opened": True})


def test_is_test_split_open(tmp_path):
    status_path = tmp_path / "status.yaml"
    status_path.write_text(yaml.safe_dump({"test_opened": True}), encoding="utf-8")
    assert is_test_split_open(load_test_open_status(status_path)) is True
