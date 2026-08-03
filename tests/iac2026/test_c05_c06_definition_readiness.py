"""C05/C06 definition readiness must stay blocked without invented evidence."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from check_c05_c06_definition_readiness import evaluate  # noqa: E402
import yaml  # noqa: E402


def test_current_definitions_block_real_run():
    defs = yaml.safe_load(
        (REPO / "reproduction/iac2026/C05_C06_DEFINITIONS.yaml").read_text(encoding="utf-8")
    )
    result = evaluate(defs, repo=REPO)
    assert result["readiness"] == "blocked"
    assert result["real_run_allowed"] is False
    assert any("real_manifest_path" in m for m in result["missing"])
    assert any("task_level" in m for m in result["missing"])
    assert any("c06_baseline_identity" in m or "pr_metric" in m for m in result["missing"])


def test_cli_exits_2_and_emits_json():
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
