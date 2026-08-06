"""ARTPS batch runner contract tests (synthetic, no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from artps_full_profile_cache import (  # noqa: E402
    PREDICTION_COLUMNS,
    build_metrics_config_snapshot,
    filter_manifest_rows,
    profile_config_sha256,
    profile_to_frozen_kwargs,
)

_filter_manifest_rows = filter_manifest_rows


@pytest.fixture
def mini_profile(tmp_path):
    profile = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/independent_eval_artps_full_frozen.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,split,binary_label,relative_path\n"
        "s1,validation,0,img/a.jpg\n"
        "s2,train,1,img/b.jpg\n"
        "s3,test,1,img/c.jpg\n",
        encoding="utf-8",
    )
    profile["dataset_manifest"] = str(manifest)
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    return profile_path, profile


def test_profile_yaml_has_required_hashes(mini_profile):
    _, profile = mini_profile
    assert profile["protocol_lock_sha256"].startswith("7767")
    assert profile["autoencoder"]["checkpoint_sha256"].startswith("8186")
    assert profile["depth"]["checkpoint_sha256"].startswith("2f21")


def test_filter_manifest_drops_test_rows_without_scoring(mini_profile):
    # Full manifests include frozen test IDs; filter must skip them, not score them.
    # Explicit test request is refused via allowed_splits / --split (see embargo tests).
    _, profile = mini_profile
    rows = [
        {"sample_id": "s1", "split": "validation", "binary_label": "0", "relative_path": "a.jpg"},
        {"sample_id": "s3", "split": "test", "binary_label": "1", "relative_path": "c.jpg"},
    ]
    filtered = _filter_manifest_rows(profile, rows)
    assert [r["sample_id"] for r in filtered] == ["s1"]
    assert all(r["split"] != "test" for r in filtered)


def test_metrics_snapshot_has_detection_fields(mini_profile):
    _, profile = mini_profile
    snap = build_metrics_config_snapshot(
        profile,
        predictions_csv=Path("pred.csv"),
        output_directory=Path("out"),
    )
    assert snap["protocol_id"] == "independent_eval_v1"
    assert snap["threshold_policy"] == "validation_selected"
    assert snap["config_id"] == "artps_full_frozen_raw_clf_on_v1"


def test_prediction_columns_match_schema():
    assert len(PREDICTION_COLUMNS) == 7
    assert "anomaly_score" in PREDICTION_COLUMNS


def test_profile_config_sha_is_stable(mini_profile):
    _, profile = mini_profile
    a = profile_config_sha256(profile)
    b = profile_config_sha256(profile)
    assert a == b
    assert len(a) == 64


def test_profile_to_frozen_kwargs_classifier_off(tmp_path):
    profile = yaml.safe_load(
        (
            REPO / "reproduction/iac2026/configs/independent_eval_artps_full_frozen_no_classifier.yaml"
        ).read_text(encoding="utf-8")
    )
    kwargs = profile_to_frozen_kwargs(profile)
    assert kwargs["enable_classifier"] is False
    assert kwargs["preprocessing_profile"] == "raw_rgb_v1"
