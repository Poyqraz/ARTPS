"""Final integrity tests for audit gate, orientation, task level, and C07 timing."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from _common import load_json_schema, sha256_file  # noqa: E402
from _config import ConfigValidationError, load_and_validate_config, load_timing_config  # noqa: E402
from audit_reproduction_inputs import audit_inputs  # noqa: E402
from cv_core_pipeline import (  # noqa: E402
    CURRENT_SURROGATE_PIPELINE_ID,
    process_frame_current_enhancement_historical_surrogate,
    process_frame_historical,
)
from detection_metrics_lib import (  # noqa: E402
    binary_auroc,
    canonical_threshold,
    confusion,
    orient_scores,
    score_orientation_meta,
)
from reproduce_detection_metrics import main as metrics_main  # noqa: E402


def _synth_cfg(tmp_path, **over):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_directory"] = str(tmp_path)
    raw.update(over)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_audit_json_missing_hashes_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = _synth_cfg(tmp_path)
    prior = {
        "passed": True,
        "evidence_mode": "software_verification",
        "claim_ids": ["C05"],
        "config_id": "example",
        "config_sha256": None,
        "manifest_sha256": None,
        "predictions_sha256": None,
        "checkpoint_sha256": None,
        "git_head": "deadbeef",
        "git_dirty": False,
        "blockers": [],
        "errors": [],
    }
    ap = tmp_path / "prior.json"
    ap.write_text(json.dumps(prior), encoding="utf-8")
    assert (
        metrics_main(
            [
                "--config",
                str(cfg),
                "--software-verification",
                "--audit-json",
                str(ap),
                "--run-id",
                "miss_hash",
            ]
        )
        == 2
    )


def test_audit_evidence_mode_mismatch_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = _synth_cfg(tmp_path)
    # Run fresh audit fields via a first successful metrics path, then mismatch mode
    assert metrics_main(["--config", str(cfg), "--software-verification", "--run-id", "base"]) == 0
    prior = json.loads((tmp_path / "base" / "input_audit.json").read_text(encoding="utf-8"))
    prior["evidence_mode"] = "real_evidence"
    ap = tmp_path / "prior.json"
    ap.write_text(json.dumps(prior), encoding="utf-8")
    assert (
        metrics_main(
            [
                "--config",
                str(cfg),
                "--software-verification",
                "--audit-json",
                str(ap),
                "--run-id",
                "mode_mismatch",
            ]
        )
        == 2
    )


def test_audit_checkpoint_hash_stale_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = _synth_cfg(tmp_path)
    assert metrics_main(["--config", str(cfg), "--software-verification", "--run-id", "base2"]) == 0
    prior = json.loads((tmp_path / "base2" / "input_audit.json").read_text(encoding="utf-8"))
    prior["checkpoint_sha256"] = "a" * 64
    ap = tmp_path / "prior.json"
    ap.write_text(json.dumps(prior), encoding="utf-8")
    assert (
        metrics_main(
            [
                "--config",
                str(cfg),
                "--software-verification",
                "--audit-json",
                str(ap),
                "--run-id",
                "ckpt_stale",
            ]
        )
        == 2
    )


def test_fixed_threshold_reversed_orientation():
    raw_t = 0.3
    canon = canonical_threshold(raw_t, False)
    assert canon == -0.3
    meta = score_orientation_meta(False)
    assert meta["canonical_score_transform"] == "negate"
    assert meta["decision_operator"] == "ge"
    y = np.array([0, 1])
    s_raw = np.array([0.9, 0.1])  # lower = more anomalous
    s = orient_scores(s_raw, False)
    assert binary_auroc(y, s) == 1.0
    y_pred = (s >= canon).astype(np.int32)
    cm = confusion(y, y_pred)
    assert cm["tp"] == 1 and cm["tn"] == 1


def test_invalid_string_positive_label_rejected(tmp_path):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["positive_label"] = "yes"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(p)


def test_prediction_config_id_mismatch_rejected(tmp_path):
    man = REPO / "reproduction/iac2026/fixtures/synthetic_manifest.csv"
    preds = tmp_path / "p.csv"
    preds.write_text(
        "sample_id,split,y_true,anomaly_score,model_name,model_version,config_id\n"
        "syn_val_0,validation,0,0.1,synth_model,v0,wrong_id\n"
        "syn_val_1,validation,1,0.9,synth_model,v0,wrong_id\n"
        "syn_test_0,test,0,0.1,synth_model,v0,wrong_id\n"
        "syn_test_1,test,1,0.9,synth_model,v0,wrong_id\n",
        encoding="utf-8",
    )
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["dataset_manifest"] = str(man)
    raw["predictions_csv"] = str(preds)
    raw["output_directory"] = str(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    loaded = load_and_validate_config(cfg_path)
    result = audit_inputs(loaded, software_verification=True)
    assert any("config_id" in e for e in result.errors)
    assert result.passed is False


def test_validation_selected_none_metric_rejected(tmp_path):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["threshold_policy"] = "validation_selected"
    raw["threshold_selection_metric"] = "none"
    raw["threshold_tie_break"] = "highest_threshold"
    raw["fixed_threshold"] = None
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(p)


def test_fixed_historical_null_threshold_rejected(tmp_path):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["fixed_threshold"] = None
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(p)


def test_pixel_binary_real_metrics_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = _synth_cfg(tmp_path, task_level="pixel_binary")
    rc = metrics_main(["--config", str(cfg), "--software-verification", "--run-id", "pix"])
    assert rc == 2


def test_region_binary_real_metrics_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = _synth_cfg(tmp_path, task_level="region_binary")
    rc = metrics_main(["--config", str(cfg), "--software-verification", "--run-id", "reg"])
    assert rc == 2


def test_invalid_timing_evidence_mode_rejected(tmp_path):
    raw = yaml.safe_load(
        (
            REPO / "reproduction/iac2026/configs/c07_software_verification.example.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["evidence_mode"] = "not_a_mode"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_timing_config(p)


def test_invalid_timing_profile_rejected(tmp_path):
    raw = yaml.safe_load(
        (
            REPO / "reproduction/iac2026/configs/c07_software_verification.example.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["profile"] = "current_production"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_timing_config(p)


def test_real_c07_images_dir_only_rejected(tmp_path):
    raw = {
        "claim_ids": ["C07"],
        "evidence_mode": "real_evidence",
        "profile": "historical_exact",
        "input_resolution": 256,
        "batch_size": 1,
        "learned_depth_enabled": False,
        "autoencoder_enabled": False,
        "warmup_count": 30,
        "timed_iteration_count": 300,
        "images_dir": str(tmp_path / "imgs"),
        "output_directory": str(tmp_path),
        "allow_dirty_git": False,
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_timing_config(p)
    # Also exercise runner if schema somehow allowed images_dir alone
    raw2 = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/c07_historical_exact.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw2.pop("input_manifest", None)
    raw2.pop("dataset_root_env", None)
    raw2["images_dir"] = str(tmp_path / "imgs")
    raw2["output_directory"] = str(tmp_path)
    p2 = tmp_path / "bad2.yaml"
    p2.write_text(yaml.safe_dump(raw2), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "benchmark_cv_core_speed.py"), "--config", str(p2)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2


def test_real_c07_manifest_hash_mismatch_rejected(tmp_path, monkeypatch):
    from benchmark_cv_core_speed import _load_manifest_frames

    img = tmp_path / "frame.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img)
    digest = sha256_file(img)
    man = tmp_path / "man.csv"
    man.write_text(
        "sample_id,relative_path,sha256,source_id,mission,instrument,sol,order_index\n"
        f"s0,frame.png,{'0'*64},src,curiosity,navcam,1,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sha256"):
        _load_manifest_frames(manifest_path=man, dataset_root=tmp_path)

    # CLI path: schema-valid real config + clean-git monkeypatch still hits mismatch
    monkeypatch.chdir(REPO)
    monkeypatch.setenv("ARTPS_DATASET_ROOT", str(tmp_path))
    import benchmark_cv_core_speed as bench

    monkeypatch.setattr(bench, "git_dirty", lambda: False)
    man2 = tmp_path / "man2.csv"
    man2.write_text(
        "sample_id,relative_path,sha256,source_id,mission,instrument,sol,order_index\n"
        f"s0,frame.png,{'0'*64},src,curiosity,navcam,1,0\n",
        encoding="utf-8",
    )
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/c07_historical_exact.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["input_manifest"] = str(man2)
    raw["dataset_root_env"] = "ARTPS_DATASET_ROOT"
    raw["output_directory"] = str(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert bench.main(["--config", str(cfg), "--run-id", "bad_sha"]) == 2
    assert digest != "0" * 64


def test_historical_resize_inside_timed_scope():
    rgb = np.zeros((384, 384, 3), dtype=np.uint8)
    rgb[10:50, 10:50] = 255
    _, _, stages = process_frame_historical(rgb, target_res=256)
    assert stages["resize_preprocess"] > 0.0
    assert "fusion_localization_combined" in stages
    assert "image_decode" not in stages


def test_no_fabricated_70_30_stage_values():
    rgb = np.random.default_rng(0).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    _, _, stages = process_frame_historical(rgb, target_res=256)
    forbidden = [k for k in stages if "0.7" in k or "0.3" in k or "seventy" in k.lower()]
    assert forbidden == []
    assert "fusion_localization_combined" in stages
    assert "fusion" not in stages or "fusion_localization_combined" in stages
    # Surrogate profile uses same combined stage name
    _, _, stages2 = process_frame_current_enhancement_historical_surrogate(rgb, target_res=256)
    assert "fusion_localization_combined" in stages2
    assert CURRENT_SURROGATE_PIPELINE_ID == "current_enhancement_historical_surrogate"


def test_timing_output_schema_validation():
    import jsonschema

    schema = load_json_schema("timing_result.schema.json")
    good = {
        "claim_ids": ["C07"],
        "evidence_class": "software_verification",
        "eligible_for_claim_closure": False,
        "input_source": "synthetic",
        "pipeline_id": "historical_opencv_surrogate_8f7e3ff",
        "profile": "historical_software_verification",
        "source_commit": "8f7e3ff",
        "implementation_hash": "a" * 64,
        "equivalence_test_status": "not_independently_verified",
        "input_manifest_sha256": None,
        "input_file_count": 8,
        "ordered_input_set_sha256": None,
        "config_sha256": "b" * 64,
        "git_head": "abc1234",
        "git_dirty": True,
        "input_resolution": 256,
        "batch_size": 1,
        "learned_depth_enabled": False,
        "autoencoder_enabled": False,
        "warmup_count": 30,
        "timed_iteration_count": 300,
        "mean_core_latency_s": 0.01,
        "mean_total_latency_s": 0.02,
        "headline_fps": 50.0,
        "headline_metric_name": "historical_exact_fps",
        "stages": {
            "frame_fetch": 0.0,
            "resize_preprocess": 0.001,
            "enhancement": 0.005,
            "reconstruction_surrogate": 0.001,
            "fallback_depth": 0.001,
            "fusion_localization_combined": 0.002,
            "core_processing": 0.009,
            "total_pipeline": 0.01,
        },
    }
    jsonschema.Draft202012Validator(schema).validate(good)
    bad = dict(good)
    bad["eligible_for_claim_closure"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(bad)


def test_equivalence_status_is_honest():
    assert CURRENT_SURROGATE_PIPELINE_ID == "current_enhancement_historical_surrogate"
    fixture = REPO / "reproduction/iac2026/fixtures/historical_core_golden.json"
    meta = json.loads(fixture.read_text(encoding="utf-8"))
    assert meta["equivalence_class"] == "regression_smoke"
    assert meta["equivalence_test_status"] == "not_independently_verified"
    schema = load_json_schema("timing_result.schema.json")
    allowed = schema["properties"]["equivalence_test_status"]["enum"]
    assert "not_independently_verified" in allowed
    assert "byte_identical_to_8f7e3ff" not in allowed
