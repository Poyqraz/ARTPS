"""Provenance / output schema tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from reproduce_detection_metrics import main as metrics_main  # noqa: E402


def test_output_schema_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/detection_reproduction.synthetic.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_directory"] = str(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert metrics_main(["--config", str(cfg), "--software-verification", "--run-id", "prov"]) == 0
    m = json.loads((tmp_path / "prov" / "detection_metrics.json").read_text(encoding="utf-8"))
    schema = json.loads(
        (REPO / "reproduction/iac2026/schemas/detection_metrics.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(m)
    prov = json.loads((tmp_path / "prov" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["accepted_abstract_targets_not_used_as_pass_fail"] is True
    assert (tmp_path / "prov" / "command.txt").is_file()
