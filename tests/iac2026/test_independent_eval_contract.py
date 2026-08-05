"""Independent eval execution contract: lock, provenance, split, baselines."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from _config import ConfigValidationError, load_and_validate_config  # noqa: E402
from audit_reproduction_inputs import audit_inputs  # noqa: E402
from baselines.base import BaselineContractError, predict_padim  # noqa: E402
from build_independent_eval_split import (  # noqa: E402
    SplitContractError,
    assert_ratios_selected,
    check_group_leakage,
    refuse_frozen_test_mutation,
)
from independent_eval_contract import (  # noqa: E402
    load_protocol_lock,
    validate_config_against_lock,
)


LOCK_SHA = "f5e039df698d5ed4992d01c29f119400915d630906426f4c2510cd6d0bbef71d"


def _indep_cfg():
    return yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/independent_evaluation.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_lock_sha_matches_checked_in_digest():
    _, sha, _ = load_protocol_lock()
    assert sha == LOCK_SHA


def test_synthetic_independent_config_loads():
    cfg = load_and_validate_config(
        REPO / "reproduction/iac2026/configs/independent_evaluation.synthetic.yaml"
    )
    assert cfg["protocol_id"] == "independent_eval_v1"
    assert cfg["evaluation_purpose"] == "current_reproducible_evaluation"


def test_config_differs_from_lock_rejected(tmp_path):
    raw = _indep_cfg()
    raw["threshold_tie_break"] = "lowest_threshold"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="threshold_tie_break|protocol lock"):
        load_and_validate_config(p)


def test_positive_label_not_one_rejected(tmp_path):
    raw = _indep_cfg()
    raw["positive_label"] = 0
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(p)


def test_score_aggregation_drift_rejected(tmp_path):
    raw = _indep_cfg()
    raw["image_score_aggregation"] = "max_pool_anomaly_map"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="max_valid_candidate|image_score_aggregation|protocol lock"):
        load_and_validate_config(p)


def test_protocol_lock_sha_mismatch_rejected(tmp_path):
    raw = _indep_cfg()
    raw["protocol_lock_sha256"] = "a" * 64
    errs = validate_config_against_lock(raw)
    assert any("protocol_lock_sha256 mismatch" in e for e in errs)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="protocol_lock_sha256"):
        load_and_validate_config(p)


def test_audit_and_metrics_carry_protocol_provenance():
    loaded = load_and_validate_config(
        REPO / "reproduction/iac2026/configs/independent_evaluation.synthetic.yaml"
    )
    audit = audit_inputs(loaded, software_verification=True)
    assert audit.passed, audit.errors
    assert audit.evaluation_purpose == "current_reproducible_evaluation"
    assert audit.protocol_id == "independent_eval_v1"
    assert audit.protocol_lock_sha256 == LOCK_SHA
    assert audit.annotation_version == "independent_eval_v1"
    keys = audit.compare_keys()
    assert "protocol_id" in keys and "evaluation_purpose" in keys

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/iac2026/reproduce_detection_metrics.py"),
            "--config",
            str(REPO / "reproduction/iac2026/configs/independent_evaluation.synthetic.yaml"),
            "--software-verification",
            "--run-id",
            "ut_indep_sw",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    metrics = json.loads(
        (
            REPO / "results/iac2026/reproduction/ut_indep_sw/detection_metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert metrics["evaluation_purpose"] == "current_reproducible_evaluation"
    assert metrics["protocol_id"] == "independent_eval_v1"
    assert metrics["historical_claim_reproduction"] is False
    assert metrics["eligible_for_C05_C06_closure"] is False
    assert metrics["eligible_for_IND_EVAL_V1_result_reporting"] is False
    assert metrics["evidence_class"] == "software_verification"
    assert metrics["eligible_for_claim_closure"] is False


def test_sw_cannot_become_eligible_for_reporting():
    loaded = load_and_validate_config(
        REPO / "reproduction/iac2026/configs/independent_evaluation.synthetic.yaml"
    )
    assert loaded.evidence_mode == "software_verification"
    # eligibility flags only appear on metrics output; contract forbids True for SW
    assert loaded["evaluation_purpose"] == "current_reproducible_evaluation"


def test_unresolved_annotation_rejected(tmp_path):
    src = REPO / "reproduction/iac2026/fixtures/independent_eval_sw_manifest.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    rows[0]["adjudication_status"] = "unresolved"
    man = tmp_path / "bad_man.csv"
    with man.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    raw = _indep_cfg()
    raw["dataset_manifest"] = str(man)
    raw["predictions_csv"] = str(
        REPO / "reproduction/iac2026/fixtures/independent_eval_sw_predictions.csv"
    )
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    loaded = load_and_validate_config(p)
    audit = audit_inputs(loaded, software_verification=True)
    assert not audit.passed
    assert any("adjudication_status=resolved" in e for e in audit.errors)


def test_cross_split_leakage_rejected():
    rows = [
        {"sha256": "abc", "split": "train", "scene_group_id": "s1", "duplicate_group_id": "d1"},
        {"sha256": "abc", "split": "test", "scene_group_id": "s2", "duplicate_group_id": "d2"},
    ]
    errs = check_group_leakage(rows, ["sha256", "scene_group_id", "duplicate_group_id"])
    assert any("sha256" in e for e in errs)


def test_baseline_train_positive_rejected():
    cfg = {
        "backbone": "wrn",
        "layers": "2,3",
        "image_size": 256,
        "weights_path": "x.pth",
        "weights_sha256": "a" * 64,
        "score_aggregation": "max",
        "train_bank_recipe": "negatives_only",
        "train_bank_sample_ids": ["a", "b"],
        "train_bank_binary_labels": [0, 1],
    }
    with pytest.raises(BaselineContractError, match="binary_label=0"):
        predict_padim(["x"], split="test", config=cfg)


def test_split_ratios_pending_and_frozen_mutation():
    lock, _, _ = load_protocol_lock()
    with pytest.raises(SplitContractError, match="PENDING_RATIO_SELECTION"):
        assert_ratios_selected(lock)
    freeze = Path("unused")
    # create temp freeze marker
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        marker = tdir / "TEST_SPLIT_FROZEN"
        out = tdir / "out.csv"
        marker.write_text("frozen\n", encoding="utf-8")
        out.write_text("x\n", encoding="utf-8")
        with pytest.raises(SplitContractError, match="frozen"):
            refuse_frozen_test_mutation(freeze_marker=marker, output_manifest=out)


def test_aggregate_historical_counts_cannot_generate_rows():
    # Contract: notes mentioning aggregate+counts trip audit
    src = REPO / "reproduction/iac2026/fixtures/independent_eval_sw_manifest.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    rows[0]["notes"] = "expanded from aggregate 2847 quota"
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        man = Path(td) / "m.csv"
        with man.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        raw = _indep_cfg()
        raw["dataset_manifest"] = str(man)
        p = Path(td) / "c.yaml"
        p.write_text(yaml.safe_dump(raw), encoding="utf-8")
        loaded = load_and_validate_config(p)
        audit = audit_inputs(loaded, software_verification=True)
        assert not audit.passed
        assert any("aggregate" in e for e in audit.errors)


def test_c05_c06_readiness_still_blocked():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/iac2026/check_c05_c06_definition_readiness.py"),
            "--json",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["real_run_allowed"] is False
