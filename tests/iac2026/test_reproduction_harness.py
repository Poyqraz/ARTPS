"""Synthetic verification for IAC 2026 reproduction harness."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "iac2026"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "baselines"))

from audit_reproduction_inputs import main as audit_main  # noqa: E402
from baselines.base import BaselineContractError  # noqa: E402
from baselines.padim_adapter import PaDiMAdapter  # noqa: E402
from baselines.patchcore_adapter import PatchCoreAdapter  # noqa: E402
from cv_core_pipeline import core_process_rgb_u8  # noqa: E402
from reproduce_detection_metrics import (  # noqa: E402
    _binary_auroc,
    _select_threshold_on_validation,
    main as metrics_main,
)


def test_schemas_exist_and_parse():
    schema_dir = REPO / "reproduction" / "iac2026" / "schemas"
    for name in (
        "dataset_manifest.schema.json",
        "prediction_table.schema.json",
        "detection_metrics.schema.json",
        "timing_result.schema.json",
    ):
        payload = json.loads((schema_dir / name).read_text(encoding="utf-8"))
        assert payload.get("$schema")


def test_example_config_blocks_real_run_via_audit(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    cfg["output_directory"] = str(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = audit_main(["--config", str(cfg_path), "--run-id", "ut_audit_block"])
    assert rc == 2
    audit = json.loads((tmp_path / "ut_audit_block" / "input_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is False
    assert any("task_level" in b for b in audit["blockers"])


def test_audit_detects_scene_leakage(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,mission,instrument,sol,source_id,source_url,relative_path,sha256,split,"
        "binary_label,label_semantics,label_source,annotation_version,scene_group_id,"
        "duplicate_group_id,notes\n"
        "a,s,s,0,a,,a.png,SYNTHETIC_a,train,0,x,x,v0,scene_shared,dup_a,n\n"
        "b,s,s,0,b,,b.png,SYNTHETIC_b,test,1,x,x,v0,scene_shared,dup_b,n\n",
        encoding="utf-8",
    )
    preds = tmp_path / "preds.csv"
    preds.write_text(
        "sample_id,split,y_true,anomaly_score,model_name,model_version,config_id\n"
        "a,train,0,0.1,m,v,c\n"
        "b,test,1,0.9,m,v,c\n",
        encoding="utf-8",
    )
    cfg = {
        "claim_ids": ["C05"],
        "task_level": "image",
        "threshold_policy": "validation_selected",
        "dataset_manifest": str(manifest),
        "predictions_csv": str(preds),
        "output_directory": str(tmp_path),
        "allow_dirty_git": True,
        "require_real_sha256": False,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = audit_main(["--config", str(cfg_path), "--run-id", "ut_leak"])
    assert rc == 2
    audit = json.loads((tmp_path / "ut_leak" / "input_audit.json").read_text(encoding="utf-8"))
    assert any("scene_group_id" in e for e in audit["errors"])


def test_audit_rejects_nan_scores(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    manifest = REPO / "reproduction/iac2026/fixtures/synthetic_manifest.csv"
    preds = tmp_path / "preds.csv"
    preds.write_text(
        "sample_id,split,y_true,anomaly_score,model_name,model_version,config_id\n"
        "syn_test_0,test,0,nan,m,v,c\n"
        "syn_test_1,test,1,0.9,m,v,c\n",
        encoding="utf-8",
    )
    cfg = {
        "claim_ids": ["C05"],
        "task_level": "image",
        "threshold_policy": "fixed_historical",
        "fixed_threshold": 0.5,
        "dataset_manifest": str(manifest),
        "predictions_csv": str(preds),
        "output_directory": str(tmp_path),
        "allow_dirty_git": True,
        "require_real_sha256": False,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = audit_main(["--config", str(cfg_path), "--run-id", "ut_nan"])
    assert rc == 2


def test_audit_rejects_single_class(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    manifest = REPO / "reproduction/iac2026/fixtures/synthetic_manifest.csv"
    preds = tmp_path / "preds.csv"
    preds.write_text(
        "sample_id,split,y_true,anomaly_score,model_name,model_version,config_id\n"
        "syn_test_0,test,0,0.1,m,v,c\n"
        "syn_test_2,test,0,0.2,m,v,c\n",
        encoding="utf-8",
    )
    cfg = {
        "claim_ids": ["C05"],
        "task_level": "image",
        "threshold_policy": "fixed_historical",
        "fixed_threshold": 0.5,
        "dataset_manifest": str(manifest),
        "predictions_csv": str(preds),
        "output_directory": str(tmp_path),
        "allow_dirty_git": True,
        "require_real_sha256": False,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = audit_main(["--config", str(cfg_path), "--run-id", "ut_single"])
    assert rc == 2


def test_metrics_no_test_threshold_search_when_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    cfg["output_directory"] = str(tmp_path)
    cfg["threshold_policy"] = "unknown"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = metrics_main(["--config", str(cfg_path), "--run-id", "ut_thr"])
    assert rc == 0
    metrics = json.loads((tmp_path / "ut_thr" / "detection_metrics.json").read_text(encoding="utf-8"))
    assert metrics["selected_threshold"] is None
    assert metrics["f1"] is None
    assert "threshold_policy_unknown" in metrics["reproduction_status"]
    # Must not compare to 0.894
    blob = (tmp_path / "ut_thr" / "detection_metrics.md").read_text(encoding="utf-8")
    assert "0.894" not in blob


def test_metrics_validation_selected_does_not_use_test_labels_for_threshold():
    y_val = np.array([0, 1], dtype=np.int32)
    s_val = np.array([0.1, 0.9], dtype=np.float64)
    t = _select_threshold_on_validation(y_val, s_val)
    # Threshold chosen only from validation scores
    assert t in (0.1, 0.9)


def test_auroc_perfect_ranking():
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    s = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    assert abs(_binary_auroc(y, s) - 1.0) < 1e-9


def test_baselines_fail_loud():
    with pytest.raises(BaselineContractError):
        PaDiMAdapter().predict_rows(["a"], split="test", config={})
    with pytest.raises(BaselineContractError):
        PatchCoreAdapter().predict_rows(["a"], split="test", config={"backbone": "WRN-50-2"})


def test_core_pipeline_runs_and_flags_in_bench(tmp_path):
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[80:120, 80:120] = 255
    combined, dets = core_process_rgb_u8(rgb)
    assert combined.shape == (256, 256)
    assert combined.dtype == np.float32 or combined.dtype == np.float64

    # Bench must reject learned depth / AE
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "benchmark_cv_core_speed.py"),
            "--allow-learned-depth",
            "--warmup",
            "30",
            "--timed",
            "300",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "learned depth" in (proc.stderr + proc.stdout).lower() or proc.returncode != 0


def test_metrics_bundle_provenance(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = yaml.safe_load(
        (
            REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml"
        ).read_text(encoding="utf-8")
    )
    cfg["output_directory"] = str(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert metrics_main(["--config", str(cfg_path), "--run-id", "ut_bundle"]) == 0
    out = tmp_path / "ut_bundle"
    for name in (
        "config_used.yaml",
        "predictions.csv",
        "detection_metrics.json",
        "environment.json",
        "provenance.json",
        "command.txt",
    ):
        assert (out / name).is_file(), name
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["accepted_abstract_targets_not_used_as_pass_fail"] is True
