"""Build model-blind annotation queue for independent_eval_v1 primary domain."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, read_csv_dicts
from independent_eval_annotation_schema import (
    ANNOTATION_QUEUE_FIELDS,
    ANNOTATION_VERSION,
    FORBIDDEN_QUEUE_COLUMNS,
    QUEUE_SEED,
)
from inventory_independent_eval_dataset import require_dataset_root


def _in_primary_domain(row: Dict[str, str]) -> bool:
    rel = row["relative_path"].replace("\\", "/")
    low = rel.lower()
    parts = rel.split("/")
    if not parts or parts[0] not in ("train", "valid"):
        return False
    if len(parts) >= 2 and parts[1].lower() in ("rover", "unlabeled"):
        return False
    # Curiosity Mastcam filename tokens (homogeneous Roboflow Mastcam-style set).
    name = row.get("filename") or Path(rel).name
    nlow = name.lower()
    if "curiosity" not in nlow:
        return False
    if "mast" not in nlow:  # MAST / mastcam
        return False
    if row.get("readable", "true").lower() != "true":
        return False
    return True


def assert_no_forbidden_columns(fieldnames: List[str]) -> None:
    low = {f.lower() for f in fieldnames}
    for bad in FORBIDDEN_QUEUE_COLUMNS:
        if bad in low or any(bad in f for f in low):
            raise ValueError(f"forbidden model-score column in annotation queue: {bad}")


def build_queue(inventory_rows: List[Dict[str, str]], seed: int = QUEUE_SEED) -> List[Dict[str, str]]:
    candidates = [r for r in inventory_rows if _in_primary_domain(r)]
    rng = random.Random(seed)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    rows: List[Dict[str, str]] = []
    for rank, idx in enumerate(order):
        r = candidates[idx]
        rows.append(
            {
                "candidate_id": r["candidate_id"],
                "relative_path": r["relative_path"],
                "raw_sha256": r["raw_sha256"],
                "mission": r.get("mission", "UNKNOWN"),
                "instrument": r.get("instrument", "UNKNOWN"),
                "source_id": r.get("source_id", "UNKNOWN"),
                "annotation_order": str(rank),
                "binary_label": "",
                "inclusion_status": "",
                "exclusion_reason": "",
                "label_confidence": "",
                "annotation_notes": "",
                "annotator_id": "",
                "annotation_timestamp": "",
                "adjudication_status": "pending",
                "annotation_version": ANNOTATION_VERSION,
            }
        )
    return rows


def write_queue(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_forbidden_columns(ANNOTATION_QUEUE_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANNOTATION_QUEUE_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ANNOTATION_QUEUE_FIELDS})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=REPO_ROOT / "results" / "iac2026" / "dataset_build" / "source_inventory.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "reproduction"
        / "iac2026"
        / "annotations"
        / "independent_eval_v1_annotation_queue.csv",
    )
    parser.add_argument("--seed", type=int, default=QUEUE_SEED)
    args = parser.parse_args(argv)
    # Fail closed if root missing (even though we only read inventory here).
    require_dataset_root()
    if not args.inventory_csv.is_file():
        print(f"inventory missing: {args.inventory_csv}", file=sys.stderr)
        return 2
    inv = read_csv_dicts(args.inventory_csv)
    rows = build_queue(inv, seed=args.seed)
    write_queue(rows, args.out)
    print(f"Wrote {len(rows)} queue rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
