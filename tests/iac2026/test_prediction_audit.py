"""Prediction-table audit tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from _config import load_and_validate_config  # noqa: E402
from audit_reproduction_inputs import audit_inputs  # noqa: E402


def _cfg(tmp_path, manifest, preds, **over):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw.update(over)
    raw["dataset_manifest"] = str(manifest)
    raw["predictions_csv"] = str(preds)
    raw["output_directory"] = str(tmp_path)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_and_validate_config(p)


def test_duplicate_prediction(tmp_path):
    man = REPO / "reproduction/iac2026/fixtures/synthetic_manifest.csv"
    preds = tmp_path / "p.csv"
    preds.write_text(
        "sample_id,split,y_true,anomaly_score,model_name,model_version,config_id\n"
        "syn_test_0,test,0,0.1,synth_model,v0,example\n"
        "syn_test_0,test,0,0.2,synth_model,v0,example\n"
        "syn_test_1,test,1,0.9,synth_model,v0,example\n",
        encoding="utf-8",
    )
    result = audit_inputs(_cfg(tmp_path, man, preds), software_verification=True)
    assert any("duplicate prediction" in e for e in result.errors)


def test_nan_inf_scores(tmp_path):
    man = REPO / "reproduction/iac2026/fixtures/synthetic_manifest.csv"
    for bad in ("nan", "inf", "-inf"):
        preds = tmp_path / f"p_{bad}.csv"
        preds.write_text(
            "sample_id,split,y_true,anomaly_score,model_name,model_version,config_id\n"
            f"syn_test_0,test,0,{bad},synth_model,v0,example\n"
            "syn_test_1,test,1,0.9,synth_model,v0,example\n",
            encoding="utf-8",
        )
        result = audit_inputs(_cfg(tmp_path, man, preds), software_verification=True)
        assert any("finite" in e or "float" in e for e in result.errors)


def test_test_only_single_class(tmp_path):
    man = REPO / "reproduction/iac2026/fixtures/synthetic_manifest.csv"
    preds = tmp_path / "p.csv"
    preds.write_text(
        "sample_id,split,y_true,anomaly_score,model_name,model_version,config_id\n"
        "syn_val_0,validation,0,0.1,synth_model,v0,example\n"
        "syn_val_1,validation,1,0.9,synth_model,v0,example\n"
        "syn_test_0,test,0,0.1,synth_model,v0,example\n"
        "syn_test_2,test,0,0.2,synth_model,v0,example\n",
        encoding="utf-8",
    )
    result = audit_inputs(_cfg(tmp_path, man, preds), software_verification=True)
    assert any("test split must contain both" in e for e in result.errors)
