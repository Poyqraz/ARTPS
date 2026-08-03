"""Detection metrics + positive label / orientation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from detection_metrics_lib import (  # noqa: E402
    binary_auroc,
    map_positive_label,
    orient_scores,
)
from reproduce_detection_metrics import main as metrics_main  # noqa: E402


def test_auroc_bounds_and_classes():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    v = binary_auroc(y, s)
    assert v is not None and 0.0 <= v <= 1.0
    assert binary_auroc(np.array([0, 0]), np.array([0.1, 0.2])) is None


def test_positive_label_mapping():
    y = np.array([2, 2, 5, 5])
    assert map_positive_label(y, 5).tolist() == [0, 0, 1, 1]


def test_reversed_score_orientation():
    y = np.array([0, 1])
    s = np.array([0.9, 0.1])  # lower = more anomalous
    assert binary_auroc(y, orient_scores(s, False)) == 1.0


def test_metrics_software_verification_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_directory"] = str(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert metrics_main(["--config", str(cfg), "--software-verification", "--run-id", "ut"]) == 0
    m = json.loads((tmp_path / "ut" / "detection_metrics.json").read_text(encoding="utf-8"))
    assert m["evidence_class"] == "software_verification"
    assert m["eligible_for_claim_closure"] is False
    assert "0.894" not in json.dumps(m)
