"""Audit gate tests for metrics runner."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from reproduce_detection_metrics import main as metrics_main  # noqa: E402


def test_metrics_requires_software_verification_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_directory"] = str(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert metrics_main(["--config", str(cfg), "--run-id", "x"]) == 2


def test_real_evidence_cannot_skip_with_software_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    # example config fails schema/policy — also rejects SW flag if somehow loaded
    assert (
        metrics_main(
            [
                "--config",
                str(REPO / "reproduction/iac2026/configs/detection_reproduction.example.yaml"),
                "--software-verification",
                "--run-id",
                "x",
            ]
        )
        == 2
    )


def test_stale_audit_hash_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_directory"] = str(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    audit = {
        "passed": True,
        "blockers": [],
        "errors": [],
        "config_sha256": "0" * 64,
        "manifest_sha256": "1" * 64,
        "predictions_sha256": "2" * 64,
        "git_head": "deadbeef",
        "git_dirty": False,
    }
    ap = tmp_path / "audit.json"
    ap.write_text(json.dumps(audit), encoding="utf-8")
    assert (
        metrics_main(
            [
                "--config",
                str(cfg),
                "--software-verification",
                "--audit-json",
                str(ap),
                "--run-id",
                "stale",
            ]
        )
        == 2
    )
