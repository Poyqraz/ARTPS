"""C07 benchmark contract tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def test_wrong_resolution_rejected(tmp_path):
    raw = yaml.safe_load(
        (
            REPO / "reproduction/iac2026/configs/c07_software_verification.example.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["input_resolution"] = 384
    raw["output_directory"] = str(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "benchmark_cv_core_speed.py"),
            "--config",
            str(cfg),
            "--software-verification",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


def test_missing_images_dir_real_rejected(tmp_path):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/c07_historical_exact.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["images_dir"] = str(tmp_path / "nope")
    raw["output_directory"] = str(tmp_path)
    raw["allow_dirty_git"] = True
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "benchmark_cv_core_speed.py"),
            "--config",
            str(cfg),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


def test_synthetic_mode_evidence_class(tmp_path):
    raw = yaml.safe_load(
        (
            REPO / "reproduction/iac2026/configs/c07_software_verification.example.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_directory"] = str(tmp_path)
    # Keep full 300 for contract; may be slow but required
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "benchmark_cv_core_speed.py"),
            "--config",
            str(cfg),
            "--software-verification",
            "--run-id",
            "ut_c07",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((tmp_path / "ut_c07" / "timing_summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_class"] == "software_verification"
    assert summary["eligible_for_claim_closure"] is False
    assert summary["input_source"] == "synthetic"
    assert summary["pipeline_id"] == "historical_opencv_surrogate_8f7e3ff"
    assert summary["equivalence_test_status"] == "not_independently_verified"
    assert summary["profile"] == "historical_software_verification"
    assert "fusion_localization_combined" in summary["stages"]
    assert "image_decode" not in summary["stages"]
    for k in summary["stages"]:
        assert "0.7" not in k and "0.3" not in k
