"""Test-split embargo helper tests."""
from __future__ import annotations

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
