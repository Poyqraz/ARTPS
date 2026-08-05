"""Dataset-only readiness audit for independent_eval_v1 (no model runs)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, read_csv_dicts, sha256_file, write_json
from build_independent_eval_split import check_group_leakage, select_split_ratios
from independent_eval_annotation_schema import ANNOTATION_VERSION, FORBIDDEN_QUEUE_COLUMNS
from independent_eval_contract import load_protocol_lock
from inventory_independent_eval_dataset import require_dataset_root


def audit_dataset_readiness(
    *,
    dataset_root: Path,
    inventory_csv: Path,
    queue_csv: Path,
    manifest_csv: Path,
    freeze_marker: Path,
    domain_doc: Path,
) -> Dict[str, Any]:
    errors: List[str] = []
    inventory_ok = inventory_csv.is_file()
    domain_ok = domain_doc.is_file()
    queue_ok = queue_csv.is_file()
    manifest_ok = manifest_csv.is_file()
    freeze_ok = freeze_marker.is_file()

    queue_rows = read_csv_dicts(queue_csv) if queue_ok else []
    if queue_ok and queue_rows:
        fields = list(queue_rows[0].keys())
        for bad in FORBIDDEN_QUEUE_COLUMNS:
            if any(bad in f.lower() for f in fields):
                errors.append(f"forbidden queue column: {bad}")

    included = []
    excluded_n = uncertain_n = 0
    if queue_ok:
        for r in queue_rows:
            if r.get("inclusion_status") == "excluded":
                excluded_n += 1
            elif r.get("inclusion_status") == "uncertain":
                uncertain_n += 1
            elif (
                r.get("inclusion_status") == "included"
                and r.get("binary_label") in ("0", "1")
                and r.get("adjudication_status") == "resolved"
                and r.get("annotation_version") == ANNOTATION_VERSION
            ):
                included.append(r)

    annotation_complete = len(included) >= 240
    # QC complete if every included row has qc note or adjudication resolved (batch QC stamps notes).
    qc_complete = annotation_complete and all(
        ("qc_" in (r.get("annotation_notes") or ""))
        or (r.get("adjudication_status") == "resolved")
        for r in included
    )

    manifest_rows = read_csv_dicts(manifest_csv) if manifest_ok else []
    hash_ok = True
    if manifest_ok and manifest_rows:
        for r in manifest_rows:
            rel = r["relative_path"].replace("\\", "/")
            path = dataset_root / rel
            if not path.is_file():
                hash_ok = False
                errors.append(f"missing file: {rel}")
                continue
            digest = sha256_file(path)
            if digest != r.get("raw_sha256") or digest != r.get("sha256"):
                hash_ok = False
                errors.append(f"sha mismatch: {r.get('sample_id')}")
                break  # fail fast
    else:
        hash_ok = False

    lock, _, _ = load_protocol_lock()
    group_fields = []
    gf = lock.get("group_fields") or {}
    if isinstance(gf, dict):
        group_fields = list(gf.get("value") or [])
    leakage = check_group_leakage(manifest_rows, group_fields) if manifest_rows else ["no_manifest"]
    leakage_ok = leakage == []

    n = len(manifest_rows) if manifest_rows else len(included)
    ratios = select_split_ratios(n)
    split_frozen = freeze_ok and ratios is not None and leakage_ok

    pos = sum(1 for r in manifest_rows if r.get("binary_label") == "1")
    neg = sum(1 for r in manifest_rows if r.get("binary_label") == "0")
    frac = (pos / n) if n else None
    balance_ok = frac is not None and 0.40 <= frac <= 0.60

    ready = all(
        [
            inventory_ok,
            domain_ok,
            annotation_complete,
            qc_complete,
            manifest_ok and len(manifest_rows) >= 240,
            split_frozen,
            hash_ok,
            leakage_ok,
            balance_ok,
            not errors,
        ]
    )

    return {
        "source_inventory_complete": inventory_ok,
        "primary_domain_locked": domain_ok,
        "annotation_complete": annotation_complete,
        "annotation_qc_complete": qc_complete,
        "manifest_complete": manifest_ok and len(manifest_rows) >= 240,
        "split_frozen": split_frozen,
        "file_hash_audit_passed": hash_ok,
        "leakage_audit_passed": leakage_ok,
        "class_balance": {
            "n": n,
            "positives": pos,
            "negatives": neg,
            "positive_fraction": frac,
            "within_40_60": balance_ok,
        },
        "included_sample_count": n,
        "excluded_sample_count": excluded_n,
        "uncertain_sample_count": uncertain_n,
        "ready_for_model_runs": ready,
        "dataset_readiness": "ready" if ready else ("insufficient" if ratios is None else "blocked"),
        "annotator_count": 1,
        "independent_double_review": False,
        "label_source": "workspace_visual_review",
        "errors": errors,
        "leakage_errors": leakage,
        "claim_support_unchanged": {
            "C05_C06_C07": "accepted_abstract_reproduction_pending",
            "IND_EVAL_V1": "protocol_defined_pending_data",
            "C08": "planned",
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "iac2026" / "dataset_build" / "dataset_readiness.json",
    )
    args = parser.parse_args(argv)
    root = require_dataset_root()
    report = audit_dataset_readiness(
        dataset_root=root,
        inventory_csv=REPO_ROOT / "results" / "iac2026" / "dataset_build" / "source_inventory.csv",
        queue_csv=REPO_ROOT
        / "reproduction"
        / "iac2026"
        / "annotations"
        / "independent_eval_v1_annotation_queue.csv",
        manifest_csv=REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1.csv",
        freeze_marker=REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "TEST_SPLIT_FROZEN",
        domain_doc=REPO_ROOT
        / "paper"
        / "iac2026"
        / "reproduction"
        / "INDEPENDENT_EVAL_V1_DOMAIN_SELECTION.md",
    )
    write_json(args.out, report)
    print(report)
    return 0 if report["ready_for_model_runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
