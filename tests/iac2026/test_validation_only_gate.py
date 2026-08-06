"""Validation-only gate tests for frozen ARTPS runners."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from artps_full_profile_cache import allowed_splits, filter_manifest_rows  # noqa: E402

_filter_manifest_rows = filter_manifest_rows


def test_allowed_splits_excludes_test_by_embargo(monkeypatch):
    monkeypatch.setattr(
        "test_split_embargo.load_test_open_status",
        lambda path=None: {"test_opened": False, "reason": "frozen"},
    )
    profile = {"allowed_splits": ["train", "validation", "test"]}
    with pytest.raises(ValueError, match="test"):
        allowed_splits(profile)


def test_runner_default_filter_skips_train(tmp_path):
    profile = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/independent_eval_artps_full_frozen.yaml").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        {"sample_id": "t", "split": "train", "binary_label": "0", "relative_path": "a.jpg"},
        {"sample_id": "v", "split": "validation", "binary_label": "1", "relative_path": "b.jpg"},
    ]
    filtered = _filter_manifest_rows(profile, rows)
    splits = {r["split"] for r in filtered}
    assert "train" in splits
    assert "validation" in splits
    validation_only = [r for r in filtered if r["split"] != "train"]
    assert len(validation_only) == 1
    assert validation_only[0]["sample_id"] == "v"


def test_profile_marks_not_final_test_result():
    profile = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/independent_eval_artps_full_frozen.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert profile.get("not_final_test_result") is True
    assert profile.get("eligible_for_manuscript_primary_results") is False
