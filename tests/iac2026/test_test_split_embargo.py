"""Test-split embargo helper tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from test_split_embargo import (  # noqa: E402
    assert_final_test_authorized,
    assert_split_allowed,
    is_test_split_open,
    load_final_test_scope,
    load_test_open_status,
)

AUTHORIZED_SCOPE = {
    "status": "authorized_for_final_test",
    "final_test_authorized": True,
    "authorization_status": {"final_test_authorized": True, "reason": "unit_test"},
}


def test_checked_in_status_is_closed():
    status = load_test_open_status()
    assert status.get("test_opened") is False


def test_final_test_scope_blocked_sanity_review():
    """Validation sanity blockers keep final test closed; historical selection preserved."""
    scope = load_final_test_scope()
    selection = json.loads(
        (REPO / "results/iac2026/independent_eval_v1/validation/profile_selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert scope["test_opened"] is False
    assert scope["final_test_authorized"] is False
    assert scope["status"] == "blocked_validation_sanity_review"
    assert scope["authorization_status"]["final_test_authorized"] is False
    frozen = scope["frozen_validation_selection"]
    assert frozen["selected_config_id"] == "artps_full_frozen_mars_clf_on_v1"
    assert frozen["precision_mode"] == "fp32"
    assert frozen["selected_threshold"] == 0.0
    assert frozen["amp_primary_allowed"] is False
    assert frozen["selected_config_id"] == selection["selected_config_id"]
    assert float(frozen["selected_threshold"]) == float(selection["selected_threshold"])
    assert frozen["selection_artifact_sha256"] == selection["artifact_sha256"]
    for key in (
        "validation_auroc_below_chance",
        "degenerate_all_positive_threshold",
        "score_orientation_not_verified",
    ):
        assert key in scope["blockers"]
    assert "open_test_split" in scope["forbidden_while_blocked"]


def test_assert_final_test_authorized_refuses_blocked_scope():
    with pytest.raises(ValueError, match="final test refused"):
        assert_final_test_authorized()


def test_assert_split_allowed_validation_ok():
    assert_split_allowed("validation", {"test_opened": False})


def test_assert_split_allowed_test_refused():
    with pytest.raises(ValueError, match="test_opened=false"):
        assert_split_allowed("test", {"test_opened": False, "reason": "embargo"})


def test_assert_split_allowed_test_open_requires_authorized_scope():
    assert_split_allowed("test", {"test_opened": True}, AUTHORIZED_SCOPE)


def test_assert_split_allowed_test_open_but_scope_blocked():
    blocked = {
        "status": "blocked_validation_sanity_review",
        "final_test_authorized": False,
        "authorization_status": {
            "final_test_authorized": False,
            "reason": "validation_sanity_review_required",
        },
    }
    with pytest.raises(ValueError, match="final test refused"):
        assert_split_allowed("test", {"test_opened": True}, blocked)


def test_is_test_split_open(tmp_path):
    status_path = tmp_path / "status.yaml"
    status_path.write_text(yaml.safe_dump({"test_opened": True}), encoding="utf-8")
    assert is_test_split_open(load_test_open_status(status_path)) is True
