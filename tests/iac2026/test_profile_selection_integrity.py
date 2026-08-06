"""Profile selection integrity: recompute from four pinned prediction CSVs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))
sys.path.insert(0, str(REPO))

from select_frozen_validation_profile import (  # noqa: E402
    ALLOWED_CONFIG_IDS,
    DEFAULT_PROFILES,
    build_selection_artifact,
)

REQUIRED_CANDIDATE_FIELDS = {
    "config_id",
    "metrics",
    "prediction_csv_sha256",
    "config_sha256",
    "checkpoint_hashes",
    "preprocessing_profile",
    "classifier_enabled",
    "precision_mode",
}
REQUIRED_METRIC_FIELDS = {
    "validation_sample_count",
    "auroc",
    "average_precision",
    "selected_threshold",
    "F1",
    "precision",
    "recall",
    "confusion_matrix",
}


def test_default_profiles_exactly_four():
    assert len(DEFAULT_PROFILES) == 4
    assert ALLOWED_CONFIG_IDS == {
        "artps_full_frozen_raw_clf_on_v1",
        "artps_full_frozen_raw_clf_off_v1",
        "artps_full_frozen_mars_clf_on_v1",
        "artps_full_frozen_mars_clf_off_v1",
    }


def test_recompute_matches_committed_selection():
    entries = [
        (cid, (REPO / pred).resolve(), (REPO / cfg).resolve())
        for cid, pred, cfg in DEFAULT_PROFILES
    ]
    for _, pred, cfg in entries:
        assert pred.is_file(), pred
        assert cfg.is_file(), cfg

    recomputed = build_selection_artifact(entries)
    committed = json.loads(
        (REPO / "results/iac2026/independent_eval_v1/validation/profile_selection.json").read_text(
            encoding="utf-8"
        )
    )

    assert recomputed["selection_split"] == "validation"
    assert recomputed["not_final_test_result"] is True
    assert recomputed["test_opened"] is False
    assert recomputed["selected_config_id"] == committed["selected_config_id"]
    assert recomputed["selected_config_id"] == "artps_full_frozen_mars_clf_on_v1"
    assert recomputed["selected_threshold"] == committed["selected_threshold"]
    assert "artifact_sha256" in recomputed and len(recomputed["artifact_sha256"]) == 64

    ids = [c["config_id"] for c in recomputed["candidates"]]
    assert set(ids) == ALLOWED_CONFIG_IDS
    assert len(ids) == 4

    for c in recomputed["candidates"]:
        assert REQUIRED_CANDIDATE_FIELDS <= set(c)
        assert REQUIRED_METRIC_FIELDS <= set(c["metrics"])
        assert c["metrics"]["validation_sample_count"] == 54
        assert c["precision_mode"] == "fp32"
        assert c["checkpoint_hashes"]["autoencoder_sha256"]
        assert c["checkpoint_hashes"]["depth_sha256"]

    # Selection rule: mars_clf_on has highest AP among the four
    by_id = {c["config_id"]: c for c in recomputed["candidates"]}
    aps = {cid: c["metrics"]["average_precision"] for cid, c in by_id.items()}
    assert aps["artps_full_frozen_mars_clf_on_v1"] == max(aps.values())


def test_rejects_extra_leaderboard_profile(tmp_path):
    entries = [
        (cid, (REPO / pred).resolve(), (REPO / cfg).resolve())
        for cid, pred, cfg in DEFAULT_PROFILES
    ]
    # Inject a fifth id by duplicating path under a fake name — should fail set equality
    bad = list(entries)
    bad.append(
        (
            "artps_full_frozen_mars_clf_on_amp_v1",
            entries[2][1],
            entries[2][2],
        )
    )
    with pytest.raises(ValueError, match="exactly four|must be exactly"):
        build_selection_artifact(bad)
