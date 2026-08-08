"""Validation artifact provenance, AMP, legacy, and test-embargo guards."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
VAL = REPO / "results/iac2026/independent_eval_v1/validation"
MANIFEST = REPO / "reproduction/iac2026/manifests/independent_eval_v1.csv"
PROTOCOL_LOCK_SHA = "7767f695746d0237803f57ffd2fef8f96a1434fca5d2f2ffaf2c799c3187dfe9"

PRIMARY_PROFILES = [
    "artps_full_frozen_raw_clf_on_v1",
    "artps_full_frozen_raw_clf_off_v1",
    "artps_full_frozen_mars_clf_on_v1",
    "artps_full_frozen_mars_clf_off_v1",
]


def _manifest_validation_ids() -> set[str]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["sample_id"] for r in rows if r.get("split") == "validation"}


def _manifest_test_ids() -> set[str]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["sample_id"] for r in rows if r.get("split") == "test"}


def test_test_open_status_closed():
    status = yaml.safe_load(
        (REPO / "reproduction/iac2026/test_freeze/TEST_OPEN_STATUS.yaml").read_text(encoding="utf-8")
    )
    assert status["test_opened"] is False
    assert status["opened_at"] is None
    assert status["selected_config_id"] is None
    assert status["reason"] == "pending_final_test_authorization"


def test_precision_parity_rejects_amp():
    payload = json.loads((VAL / "precision_parity.json").read_text(encoding="utf-8"))
    assert payload["decision"] == "reject_amp_keep_fp32"
    assert payload["passed"] is False
    assert payload["primary_precision"] == "fp32"
    assert payload["split"] == "validation"
    assert payload["gates"]["score_abs_tolerance"] == 0.0001
    assert payload["gates"]["metric_abs_tolerance"] == 0.0001
    assert payload["errors"]
    assert all("test" not in e.lower() or "validation" in e.lower() for e in payload["errors"])


def test_primary_prediction_csvs_match_validation_manifest():
    val_ids = _manifest_validation_ids()
    test_ids = _manifest_test_ids()
    assert len(val_ids) == 54
    for cid in PRIMARY_PROFILES:
        path = VAL / cid / "predictions.csv"
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 54
        ids = [r["sample_id"] for r in rows]
        assert len(ids) == len(set(ids))
        assert set(ids) == val_ids
        assert not (set(ids) & test_ids)
        assert {r["split"] for r in rows} == {"validation"}
        assert {r["config_id"] for r in rows} == {cid}
        for r in rows:
            score = float(r["anomaly_score"])
            assert math.isfinite(score)


def test_no_test_split_in_validation_artifacts():
    test_ids = _manifest_test_ids()
    for path in VAL.rglob("predictions.csv"):
        text = path.read_text(encoding="utf-8")
        assert ",test," not in text
        assert "\ntest," not in text
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r.get("split") != "test"
            assert r["sample_id"] not in test_ids


def test_legacy_exploratory_labels():
    for name in ("padim_legacy_exploratory", "patchcore_legacy_exploratory"):
        prov = json.loads(
            (VAL / name / "legacy_exploratory" / "provenance.json").read_text(encoding="utf-8")
        )
        assert prov["evaluation_role"] == "secondary_exploratory"
        assert prov["training_provenance"] == "unverified"
        assert prov["eligible_for_primary_baseline_table"] is False
        assert prov["eligible_for_C06_reproduction"] is False
        assert prov["eligible_for_claim_closure"] is False
        with (VAL / name / "legacy_exploratory" / "predictions.csv").open(
            "r", encoding="utf-8", newline=""
        ) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 54
        assert all(r["split"] == "validation" for r in rows)


def test_primary_provenance_links():
    for cid in PRIMARY_PROFILES:
        prov = json.loads((VAL / cid / "provenance.json").read_text(encoding="utf-8"))
        assert prov.get("predictions_sha256")
        assert prov.get("config_sha256")
        assert prov.get("manifest_sha256")
        assert prov.get("not_final_test_result") is True
        env_path = VAL / cid / "environment.json"
        assert env_path.is_file()
        run_cfg = VAL / cid / "run_config.snapshot.yaml"
        assert run_cfg.is_file()
        snap = yaml.safe_load(run_cfg.read_text(encoding="utf-8"))
        assert snap.get("protocol_lock_sha256") == PROTOCOL_LOCK_SHA


def test_claim_support_levels_unchanged():
    text = (REPO / "paper/iac2026/CLAIM_EVIDENCE_LEDGER.md").read_text(encoding="utf-8")
    assert "| C05 |" in text and "accepted_abstract_reproduction_pending" in text
    assert "| C06 |" in text and "accepted_abstract_reproduction_pending" in text
    assert "| C07 |" in text and "accepted_abstract_reproduction_pending" in text
    assert "| IND_EVAL_V1 |" in text and "protocol_defined_pending_data" in text
    # Must not claim measured on IND_EVAL_V1 (do not match IND_EVAL_V1_1).
    for line in text.splitlines():
        if line.startswith("| IND_EVAL_V1 |") and not line.startswith("| IND_EVAL_V1_1 |"):
            assert "measured" not in line.lower() or "not measured" in line.lower() or "pending" in line
