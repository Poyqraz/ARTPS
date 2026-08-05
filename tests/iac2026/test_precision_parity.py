"""Precision parity gate tests (synthetic)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from check_precision_parity import METRIC_ABS_TOL, SCORE_ABS_TOL, main  # noqa: E402


def _write_preds(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "split",
                "y_true",
                "anomaly_score",
                "model_name",
                "model_version",
                "config_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_parity_passes_within_gate(tmp_path):
    fp32 = tmp_path / "fp32.csv"
    amp = tmp_path / "amp.csv"
    rows = [
        {
            "sample_id": "a",
            "split": "validation",
            "y_true": 0,
            "anomaly_score": 0.5,
            "model_name": "m",
            "model_version": "v",
            "config_id": "c",
        },
        {
            "sample_id": "b",
            "split": "validation",
            "y_true": 1,
            "anomaly_score": 0.9,
            "model_name": "m",
            "model_version": "v",
            "config_id": "c",
        },
    ]
    _write_preds(fp32, rows)
    _write_preds(amp, rows)
    rc = main(["--fp32", str(fp32), "--amp", str(amp), "--output", str(tmp_path / "report.json")])
    assert rc == 0


def test_parity_fails_without_fp32_reference(tmp_path):
    amp = tmp_path / "amp.csv"
    _write_preds(
        amp,
        [
            {
                "sample_id": "a",
                "split": "validation",
                "y_true": 0,
                "anomaly_score": 0.5,
                "model_name": "m",
                "model_version": "v",
                "config_id": "c",
            }
        ],
    )
    rc = main(["--fp32", str(tmp_path / "missing.csv"), "--amp", str(amp)])
    assert rc == 2


def test_parity_fails_score_drift(tmp_path):
    fp32 = tmp_path / "fp32.csv"
    amp = tmp_path / "amp.csv"
    base = {
        "sample_id": "a",
        "split": "validation",
        "y_true": 0,
        "anomaly_score": 0.5,
        "model_name": "m",
        "model_version": "v",
        "config_id": "c",
    }
    _write_preds(fp32, [base])
    drift = dict(base)
    drift["anomaly_score"] = 0.5 + SCORE_ABS_TOL * 10
    _write_preds(amp, [drift])
    rc = main(["--fp32", str(fp32), "--amp", str(amp), "--output", str(tmp_path / "bad.json")])
    assert rc == 2


def test_gate_constants_not_loosened():
    assert SCORE_ABS_TOL == 1e-4
    assert METRIC_ABS_TOL == 1e-4
