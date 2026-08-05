"""Legacy exploratory baseline contract tests (synthetic)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from run_legacy_baseline_exploratory import _load_config, _require_contract  # noqa: E402


def test_padim_config_has_exploratory_role():
    cfg = _load_config(
        REPO / "reproduction/iac2026/configs/independent_eval_padim_legacy_exploratory.yaml"
    )
    errors = _require_contract(cfg)
    assert errors == []
    assert cfg["eligible_for_primary_baseline_table"] is False
    assert cfg["score_aggregation"] == "max_anomaly_map_exploratory"


def test_missing_contract_keys_fail_loud(tmp_path):
    cfg = {
        "evaluation_role": "secondary_exploratory",
        "score_aggregation": "max_anomaly_map_exploratory",
        "baseline_type": "padim",
    }
    errors = _require_contract(cfg)
    assert any("missing contract keys" in e for e in errors)


def test_wrong_evaluation_role_rejected():
    cfg = yaml.safe_load(
        (
            REPO / "reproduction/iac2026/configs/independent_eval_padim_legacy_exploratory.yaml"
        ).read_text(encoding="utf-8")
    )
    cfg["evaluation_role"] = "primary"
    errors = _require_contract(cfg)
    assert any("secondary_exploratory" in e for e in errors)


def test_patchcore_config_loads():
    cfg = _load_config(
        REPO / "reproduction/iac2026/configs/independent_eval_patchcore_legacy_exploratory.yaml"
    )
    assert cfg["baseline_type"] == "patchcore"
    assert cfg["checkpoint_sha256"].startswith("73dd")
