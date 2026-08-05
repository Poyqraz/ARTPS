"""Dataset readiness and annotator import hygiene."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from audit_independent_eval_dataset_readiness import audit_dataset_readiness  # noqa: E402


def test_annotator_does_not_import_inference():
    path = REPO / "scripts" / "iac2026" / "annotate_independent_eval_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    banned = ("torch", "torchvision", "src.models", "baselines")
    for name in imports:
        assert not any(name == b or name.startswith(b + ".") for b in banned), name


def test_readiness_json_schema_keys_when_present():
    path = REPO / "results" / "iac2026" / "dataset_build" / "dataset_readiness.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "source_inventory_complete",
        "primary_domain_locked",
        "annotation_complete",
        "annotation_qc_complete",
        "manifest_complete",
        "split_frozen",
        "file_hash_audit_passed",
        "leakage_audit_passed",
        "class_balance",
        "included_sample_count",
        "excluded_sample_count",
        "uncertain_sample_count",
        "ready_for_model_runs",
    ):
        assert key in data
    assert data["claim_support_unchanged"]["IND_EVAL_V1"] == "protocol_defined_pending_data"
