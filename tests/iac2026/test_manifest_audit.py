"""Manifest audit tests."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from _config import load_and_validate_config  # noqa: E402
from audit_reproduction_inputs import audit_inputs  # noqa: E402
from _common import resolve_under_dataset_root  # noqa: E402


def _synth_cfg(tmp_path, **overrides):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw.update(overrides)
    raw["output_directory"] = str(tmp_path)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_and_validate_config(p)


def test_path_traversal_rejected(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(ValueError, match="traversal|absolute|escapes"):
        resolve_under_dataset_root(root, "../secret.png")
    with pytest.raises(ValueError):
        resolve_under_dataset_root(root, str(tmp_path / "abs.png"))


def test_duplicate_sha_across_split(tmp_path):
    manifest = tmp_path / "m.csv"
    digest = "a" * 64
    manifest.write_text(
        "sample_id,mission,instrument,sol,source_id,source_url,relative_path,sha256,split,"
        "binary_label,label_semantics,label_source,annotation_version,scene_group_id,"
        "duplicate_group_id,notes\n"
        f"a,s,s,0,s0,,a.png,{digest},train,0,x,x,v0,sc_a,d_a,n\n"
        f"b,s,s,0,s1,,b.png,{digest},test,1,x,x,v0,sc_b,d_b,n\n",
        encoding="utf-8",
    )
    preds = tmp_path / "p.csv"
    preds.write_text(
        "sample_id,split,y_true,anomaly_score,model_name,model_version,config_id\n"
        "a,train,0,0.1,synth_model,v0,example\n"
        "b,test,1,0.9,synth_model,v0,example\n"
        "b2,test,0,0.2,synth_model,v0,example\n",
        encoding="utf-8",
    )
    # need both classes on test — add another row to manifest
    manifest.write_text(
        "sample_id,mission,instrument,sol,source_id,source_url,relative_path,sha256,split,"
        "binary_label,label_semantics,label_source,annotation_version,scene_group_id,"
        "duplicate_group_id,notes\n"
        f"a,s,s,0,s0,,a.png,{digest},train,0,x,x,v0,sc_a,d_a,n\n"
        f"b,s,s,0,s1,,b.png,{digest},test,1,x,x,v0,sc_b,d_b,n\n"
        f"c,s,s,0,s2,,c.png,{'b'*64},test,0,x,x,v0,sc_c,d_c,n\n",
        encoding="utf-8",
    )
    preds.write_text(
        "sample_id,split,y_true,anomaly_score,model_name,model_version,config_id\n"
        "b,test,1,0.9,synth_model,v0,example\n"
        "c,test,0,0.2,synth_model,v0,example\n",
        encoding="utf-8",
    )
    cfg = _synth_cfg(
        tmp_path,
        dataset_manifest=str(manifest),
        predictions_csv=str(preds),
        require_real_sha256=False,
    )
    result = audit_inputs(cfg, software_verification=True)
    assert any("sha256" in e and "spans" in e for e in result.errors)


def test_real_missing_file_and_sha_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    img = root / "ok.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    digest = hashlib.sha256(img.read_bytes()).hexdigest()
    wrong = "0" * 64
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "sample_id,mission,instrument,sol,source_id,source_url,relative_path,sha256,split,"
        "binary_label,label_semantics,label_source,annotation_version,scene_group_id,"
        "duplicate_group_id,notes\n"
        f"a,s,s,0,s0,,ok.png,{wrong},test,1,x,x,v0,sc_a,d_a,n\n"
        f"b,s,s,0,s1,,missing.png,{digest},test,0,x,x,v0,sc_b,d_b,n\n",
        encoding="utf-8",
    )
    preds = tmp_path / "p.csv"
    preds.write_text(
        "sample_id,split,y_true,anomaly_score,model_name,model_version,config_id\n"
        "a,test,1,0.9,synth_model,v0,example\n"
        "b,test,0,0.1,synth_model,v0,example\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARTPS_DATASET_ROOT", str(root))
    cfg = _synth_cfg(
        tmp_path,
        dataset_manifest=str(manifest),
        predictions_csv=str(preds),
        require_real_sha256=True,
        evidence_mode="software_verification",
    )
    # require_real_sha256 triggers file checks even in SW mode
    result = audit_inputs(cfg, software_verification=True)
    assert any("sha256 mismatch" in e for e in result.errors)
    assert any("missing file" in e for e in result.errors)
