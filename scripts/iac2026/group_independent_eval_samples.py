"""Exact-SHA and perceptual near-duplicate / scene grouping for independent_eval_v1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, write_json
from inventory_independent_eval_dataset import require_dataset_root


def average_hash(path: Path, hash_size: int = 8) -> str:
    with Image.open(path) as im:
        im = im.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float64)
        mean = float(arr.mean())
        bits = (arr > mean).astype(np.uint8).flatten()
    # Pack bits to hex
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def hamming(a: str, b: str) -> int:
    x = int(a, 16) ^ int(b, 16)
    return int(x.bit_count())


def group_rows(
    rows: List[Dict[str, str]],
    dataset_root: Path,
    *,
    phash_threshold: int = 5,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    by_sha: Dict[str, List[int]] = defaultdict(list)
    phashes: List[Optional[str]] = []
    for i, r in enumerate(rows):
        by_sha[r["raw_sha256"]].append(i)
        rel = r["relative_path"].replace("\\", "/")
        path = dataset_root / rel
        if path.is_file():
            try:
                phashes.append(average_hash(path))
            except Exception:  # noqa: BLE001
                phashes.append(None)
        else:
            phashes.append(None)

    dup_ids = {sha: f"dup_{sha[:16]}" for sha in by_sha}
    # Union-find for perceptual near-dups among different SHAs.
    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for sha, idxs in by_sha.items():
        for a, b in zip(idxs, idxs[1:]):
            union(a, b)

    # Perceptual matches are assistive only — report, do not auto-merge into hard groups.
    auto_matches: List[Dict[str, str]] = []
    for i in range(len(rows)):
        if phashes[i] is None:
            continue
        for j in range(i + 1, len(rows)):
            if phashes[j] is None:
                continue
            if rows[i]["raw_sha256"] == rows[j]["raw_sha256"]:
                continue
            dist = hamming(phashes[i], phashes[j])
            if dist <= phash_threshold:
                auto_matches.append(
                    {
                        "a": rows[i]["candidate_id"],
                        "b": rows[j]["candidate_id"],
                        "hamming": str(dist),
                        "needs_manual_spot_check": "true",
                    }
                )

    # Also union rows that share a known source_id (same product family).
    by_src: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        src = (r.get("source_id") or "").strip()
        if src and src != "UNKNOWN":
            by_src[src].append(i)
    for idxs in by_src.values():
        for a, b in zip(idxs, idxs[1:]):
            union(a, b)

    # Scene groups: same source_id when known; else duplicate-group root; else singleton.
    scene_for_idx: Dict[int, str] = {}
    for i, r in enumerate(rows):
        root_i = find(i)
        dup_id = dup_ids[r["raw_sha256"]]
        # Merge perceptual component into duplicate id using root index SHA.
        root_sha = rows[root_i]["raw_sha256"]
        dup_id = f"dup_{root_sha[:16]}"
        src = (r.get("source_id") or "UNKNOWN").strip()
        if src and src != "UNKNOWN":
            scene = f"scene_src_{src}"
        else:
            scene = f"scene_{dup_id}"
        scene_for_idx[i] = scene
        r["duplicate_group_id"] = dup_id
        r["scene_group_id"] = scene
        r["phash"] = phashes[i] or ""

    report = {
        "n_rows": len(rows),
        "n_exact_sha_groups": len(by_sha),
        "n_multi_sha_groups": sum(1 for v in by_sha.values() if len(v) > 1),
        "n_perceptual_auto_matches": len(auto_matches),
        "phash_threshold": phash_threshold,
        "perceptual_auto_matches": auto_matches[:200],
        "note": "Perceptual matches are assistive; spot-check before trusting as hard GT.",
    }
    return rows, report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=REPO_ROOT
        / "reproduction"
        / "iac2026"
        / "annotations"
        / "independent_eval_v1_annotation_queue.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO_ROOT / "results" / "iac2026" / "dataset_build" / "grouped_annotation_rows.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "results" / "iac2026" / "dataset_build" / "grouping_report.json",
    )
    parser.add_argument("--phash-threshold", type=int, default=5)
    args = parser.parse_args(argv)
    root = require_dataset_root()
    with args.queue.open("r", encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))
    # Group primary-eval candidates (included) plus keep others with singleton groups.
    focus = [
        r
        for r in all_rows
        if r.get("inclusion_status") == "included"
        and r.get("adjudication_status") == "resolved"
        and r.get("binary_label") in ("0", "1")
    ]
    others = [r for r in all_rows if r not in focus]
    focus, report = group_rows(focus, root, phash_threshold=args.phash_threshold)
    for r in others:
        sha = r.get("raw_sha256") or hashlib.sha256(r.get("candidate_id", "").encode()).hexdigest()
        r["duplicate_group_id"] = f"dup_{sha[:16]}"
        src = (r.get("source_id") or "UNKNOWN").strip()
        r["scene_group_id"] = f"scene_src_{src}" if src != "UNKNOWN" else f"scene_{r['duplicate_group_id']}"
        r["phash"] = ""
    rows = focus + others
    report["n_non_primary_rows"] = len(others)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    for extra in ("duplicate_group_id", "scene_group_id", "phash"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    write_json(args.report, report)
    print(f"grouped {len(rows)} rows; perceptual_auto_matches={report['n_perceptual_auto_matches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
