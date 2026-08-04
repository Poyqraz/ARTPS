"""C05/C06 definition readiness: attestation metadata never unlocks alone."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from check_c05_c06_definition_readiness import evaluate  # noqa: E402


def _base_defs():
    return yaml.safe_load(
        (REPO / "reproduction/iac2026/C05_C06_DEFINITIONS.yaml").read_text(encoding="utf-8")
    )


def _closed_node(value):
    return {
        "value": value,
        "status": "LOCATED",
        "required_for_real_run": True,
        "sources": [{"file": "tests/synthetic"}],
    }


def test_current_definitions_block_real_run():
    defs = _base_defs()
    result = evaluate(defs, repo=REPO)
    assert result["readiness"] == "blocked"
    assert result["real_run_allowed"] is False
    assert result["author_attestation_status"] == "pending"
    assert result["author_verified"] is False
    assert any("real_manifest_path" in m for m in result["missing"])
    assert any("task_level" in m for m in result["missing"])
    assert any("c06_baseline_identity" in m or "pr_metric" in m for m in result["missing"])


def test_cli_exits_2_and_emits_attestation_fields():
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
    assert payload["readiness"] == "blocked"
    assert payload["author_attestation_status"] == "pending"
    assert payload["author_verified"] is False


def test_audit_observation_is_not_author_attestation():
    defs = _base_defs()
    assert defs["response_status"]["type"] == "audit_observation"
    assert defs["response_status"]["author_verified"] is False
    result = evaluate(defs, repo=REPO)
    assert result["author_verified"] is False
    assert result["real_run_allowed"] is False


def test_author_verified_true_without_artifacts_still_blocked():
    defs = copy.deepcopy(_base_defs())
    defs["author_attestation"] = {
        "path": "paper/iac2026/reproduction/AUTHOR_ATTESTATION_C05_C06.template.md",
        "status": "provided",
        "author_verified": True,
    }
    result = evaluate(defs, repo=REPO)
    assert result["author_verified"] is True
    assert result["author_attestation_status"] == "provided"
    assert result["readiness"] == "blocked"
    assert result["real_run_allowed"] is False
    assert any("task_level" in m for m in result["missing"])
    assert any("real_manifest_path" in m for m in result["missing"])


def test_evidence_closed_defs_allow_ready(tmp_path):
    defs = copy.deepcopy(_base_defs())
    # Minimal closed P0 set
    defs["task_level"] = _closed_node("image_binary")
    defs["positive_class"] = _closed_node(1)
    defs["anomaly_score_definition"] = _closed_node("image_anomaly_score")
    defs["score_orientation"] = _closed_node("higher_is_more_anomalous")
    defs["sample_unit"] = _closed_node("image")
    defs["train_split"] = _closed_node("train")
    defs["validation_split"] = _closed_node("validation")
    defs["test_split"] = _closed_node("test")
    defs["label_semantics"] = _closed_node("anomaly_binary")
    defs["threshold_policy"] = _closed_node("validation_selected")
    defs["threshold_selection_metric"] = _closed_node("f1")
    defs["pr_metric_method"] = _closed_node("average_precision")
    defs["c06_baseline_identity"] = _closed_node("padim_only")
    defs["raw_predictions"] = _closed_node("results/fake_predictions.csv")
    defs["re_inference_path_documented"] = {
        "value": None,
        "status": "UNKNOWN_EVIDENCE_NOT_LOCATED",
        "required_for_real_run": True,
        "sources": [],
    }
    man = tmp_path / "real_manifest.csv"
    man.write_text(
        "sample_id,split,binary_label\n"
        "s0,train,0\n"
        "s1,validation,1\n"
        "s2,test,0\n",
        encoding="utf-8",
    )
    defs["real_manifest_path"] = str(man)
    defs["author_attestation"] = {
        "path": "paper/iac2026/reproduction/AUTHOR_ATTESTATION_C05_C06.template.md",
        "status": "pending",
        "author_verified": False,
    }
    result = evaluate(defs, repo=REPO)
    assert result["readiness"] == "ready"
    assert result["real_run_allowed"] is True
    assert result["author_verified"] is False  # attestation not required to be true once artifacts closed
    assert result["missing"] == []
