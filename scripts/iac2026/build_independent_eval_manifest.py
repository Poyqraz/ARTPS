"""Build SHA-pinned independent_eval_v1.csv from annotated + grouped rows."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, read_csv_dicts
from build_independent_eval_split import assert_no_aggregate_quota
from independent_eval_annotation_schema import ANNOTATION_VERSION, LABEL_SOURCE
from inventory_independent_eval_dataset import require_dataset_root

MANIFEST_FIELDS = [
    "sample_id",
    "mission",
    "instrument",
    "sol",
    "source_id",
    "source_url",
    "product_id",
    "sequence_id",
    "acquisition_timestamp",
    "relative_path",
    "sha256",
    "raw_sha256",
    "derived_sha256",
    "preprocessing_version",
    "split",
    "binary_label",
    "label_semantics",
    "label_source",
    "annotation_version",
    "annotator_count",
    "label_confidence",
    "adjudication_status",
    "inclusion_status",
    "exclusion_reason",
    "scene_group_id",
    "duplicate_group_id",
    "notes",
]


def row_to_manifest(r: Dict[str, str], *, split: str = "train") -> Dict[str, str]:
    sha = r["raw_sha256"]
    return {
        "sample_id": r["candidate_id"],
        "mission": r.get("mission") or "UNKNOWN",
        "instrument": r.get("instrument") or "UNKNOWN",
        "sol": "UNKNOWN",
        "source_id": r.get("source_id") or "UNKNOWN",
        "source_url": "",
        "product_id": r.get("source_id") or "UNKNOWN",
        "sequence_id": "UNKNOWN",
        "acquisition_timestamp": "",
        "relative_path": r["relative_path"].replace("\\", "/"),
        "sha256": sha,
        "raw_sha256": sha,
        "derived_sha256": sha,
        "preprocessing_version": "raw_v1",
        "split": split,
        "binary_label": r["binary_label"],
        "label_semantics": "anomaly_binary",
        "label_source": LABEL_SOURCE,
        "annotation_version": ANNOTATION_VERSION,
        "annotator_count": "1",
        "label_confidence": r.get("label_confidence") or "medium",
        "adjudication_status": r.get("adjudication_status") or "resolved",
        "inclusion_status": r.get("inclusion_status") or "included",
        "exclusion_reason": r.get("exclusion_reason") or "",
        "scene_group_id": r.get("scene_group_id") or f"scene_{sha[:16]}",
        "duplicate_group_id": r.get("duplicate_group_id") or f"dup_{sha[:16]}",
        "notes": r.get("annotation_notes") or "",
    }


def select_primary_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        assert_no_aggregate_quota(str(r.get("annotation_notes") or "") + " " + str(r.get("candidate_id") or ""))
        if r.get("inclusion_status") != "included":
            continue
        if r.get("binary_label") not in ("0", "1"):
            continue
        if r.get("adjudication_status") != "resolved":
            continue
        if r.get("annotation_version") != ANNOTATION_VERSION:
            continue
        out.append(r)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grouped-csv",
        type=Path,
        default=REPO_ROOT / "results" / "iac2026" / "dataset_build" / "grouped_annotation_rows.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1.csv",
    )
    args = parser.parse_args(argv)
    require_dataset_root()
    assert_no_aggregate_quota(str(args.grouped_csv))
    rows = read_csv_dicts(args.grouped_csv)
    primary = select_primary_rows(rows)
    # Placeholder split=train until freeze builder assigns; schema requires a valid enum.
    manifest = [row_to_manifest(r, split="train") for r in primary]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in manifest:
            w.writerow(row)
    print(f"Wrote {len(manifest)} manifest rows -> {args.out}")
    if len(manifest) < 240:
        print("insufficient included N < 240", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
