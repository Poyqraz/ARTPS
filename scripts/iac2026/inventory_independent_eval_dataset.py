"""Inventory real Mars images under ARTPS_DATASET_ROOT for independent_eval_v1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, write_json, write_text

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

INVENTORY_FIELDS = [
    "candidate_id",
    "relative_path",
    "filename",
    "extension",
    "width",
    "height",
    "channels",
    "file_size",
    "raw_sha256",
    "mission",
    "instrument",
    "sol",
    "source_id",
    "product_id",
    "sequence_id",
    "source_metadata_status",
    "readable",
    "quality_flags",
    "duplicate_candidate_group",
    "notes",
]

DATASET_ROOT_MSG = (
    "DATASET ROOT REQUIRED:\n"
    "Set ARTPS_DATASET_ROOT to the folder containing the real candidate Mars images."
)


def require_dataset_root(env_name: str = "ARTPS_DATASET_ROOT") -> Path:
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        print(DATASET_ROOT_MSG, file=sys.stderr)
        raise SystemExit(2)
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        print(DATASET_ROOT_MSG, file=sys.stderr)
        print(f"(path missing or not a directory: {root})", file=sys.stderr)
        raise SystemExit(2)
    return root


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_id(relative_path: str, raw_sha256: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", relative_path.replace("\\", "/"))
    return f"ie1_{stem[:80]}_{raw_sha256[:12]}"


def parse_metadata(relative_path: str, filename: str) -> Dict[str, str]:
    """Extract metadata only when certain; otherwise UNKNOWN (no guessing)."""
    low = f"{relative_path}/{filename}".lower().replace("\\", "/")
    mission = "UNKNOWN"
    instrument = "UNKNOWN"
    # Explicit filename tokens only (Curiosity Mastcam Roboflow exports).
    if re.search(r"curiosity", low):
        mission = "Curiosity"
    if re.search(r"\bmastcam[-_]?z\b", low) or "mastcam-z" in low:
        instrument = "Mastcam-Z"
    elif re.search(r"\bmastcam\b", low) or re.search(r"_mast_", low):
        instrument = "Mastcam"

    sol = "UNKNOWN"
    sol_m = re.search(r"(?:^|[_\-./])sol[_-]?(\d{1,5})(?:[_\-./]|$)", low)
    if sol_m:
        sol = sol_m.group(1)

    product_id = "UNKNOWN"
    # Roboflow-style curiosity_NNN_MAST_... stem before .rf.
    prod_m = re.search(r"(curiosity_\d+_mast_[a-z0-9]+)", filename.lower())
    if prod_m:
        product_id = prod_m.group(1)

    source_id = product_id if product_id != "UNKNOWN" else "UNKNOWN"
    sequence_id = "UNKNOWN"
    seq_m = re.search(r"(?:seq|sequence)[_-]?([a-z0-9]+)", low)
    if seq_m:
        sequence_id = seq_m.group(1)

    fields = {
        "mission": mission,
        "instrument": instrument,
        "sol": sol,
        "source_id": source_id,
        "product_id": product_id,
        "sequence_id": sequence_id,
    }
    known = sum(1 for v in fields.values() if v != "UNKNOWN")
    if known == 0:
        status = "none"
    elif known < len(fields):
        status = "partial"
    else:
        status = "complete"
    fields["source_metadata_status"] = status
    return fields


def read_image_meta(path: Path) -> Tuple[bool, Optional[int], Optional[int], Optional[int], List[str], str]:
    flags: List[str] = []
    notes = ""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            w, h = im.size
            mode = im.mode
            channels = {"L": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}.get(mode, len(mode))
            if min(w, h) < 64:
                flags.append("tiny_resolution")
            if mode not in ("RGB", "L", "RGBA"):
                flags.append(f"unusual_mode_{mode}")
            return True, w, h, channels, flags, notes
    except Exception as exc:  # noqa: BLE001 — inventory must flag any open failure
        return False, None, None, None, ["unreadable"], f"open_failed:{type(exc).__name__}"


def iter_image_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(Path(dirpath) / name)
    files.sort(key=lambda p: str(p).replace("\\", "/").lower())
    return files


def build_inventory(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sha_groups: Dict[str, List[str]] = {}
    for path in iter_image_files(root):
        rel = str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
        raw_sha = sha256_file(path)
        readable, w, h, ch, flags, notes = read_image_meta(path)
        meta = parse_metadata(rel, path.name)
        # Folder hint only (not a label): record terrain folder if present.
        parts = rel.split("/")
        if len(parts) >= 2 and parts[0] in ("train", "valid") and len(parts) >= 2:
            folder = parts[1]
            if folder.lower() == "rover":
                flags.append("path_hint_rover_hardware")
            flags.append(f"path_folder_{folder}")
        cid = candidate_id(rel, raw_sha)
        sha_groups.setdefault(raw_sha, []).append(cid)
        rows.append(
            {
                "candidate_id": cid,
                "relative_path": rel,
                "filename": path.name,
                "extension": path.suffix.lower(),
                "width": "" if w is None else str(w),
                "height": "" if h is None else str(h),
                "channels": "" if ch is None else str(ch),
                "file_size": str(path.stat().st_size),
                "raw_sha256": raw_sha,
                "mission": meta["mission"],
                "instrument": meta["instrument"],
                "sol": meta["sol"],
                "source_id": meta["source_id"],
                "product_id": meta["product_id"],
                "sequence_id": meta["sequence_id"],
                "source_metadata_status": meta["source_metadata_status"],
                "readable": "true" if readable else "false",
                "quality_flags": "|".join(flags),
                "duplicate_candidate_group": "",  # filled below
                "notes": notes,
            }
        )
    # Exact-SHA duplicate groups
    for row in rows:
        members = sha_groups[row["raw_sha256"]]
        if len(members) > 1:
            row["duplicate_candidate_group"] = f"sha_{row['raw_sha256'][:16]}"
            if "exact_sha_duplicate" not in row["quality_flags"]:
                flags = [f for f in row["quality_flags"].split("|") if f]
                flags.append("exact_sha_duplicate")
                row["quality_flags"] = "|".join(flags)
        else:
            row["duplicate_candidate_group"] = f"sha_{row['raw_sha256'][:16]}"
    return rows


def write_inventory(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "source_inventory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in INVENTORY_FIELDS})

    readable_n = sum(1 for r in rows if r["readable"] == "true")
    dup_n = sum(1 for r in rows if "exact_sha_duplicate" in r["quality_flags"])
    mission_c = Counter(r["mission"] for r in rows)
    instrument_c = Counter(r["instrument"] for r in rows)
    payload = {
        "dataset_root_env": "ARTPS_DATASET_ROOT",
        "n_files": len(rows),
        "n_readable": readable_n,
        "n_unreadable": len(rows) - readable_n,
        "n_exact_sha_duplicate_members": dup_n,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "mission_counts": dict(mission_c),
        "instrument_counts": dict(instrument_c),
        "rows": rows,
    }
    write_json(out_dir / "source_inventory.json", payload)

    md = [
        "# Source inventory (independent_eval_v1)",
        "",
        f"- Files: **{len(rows)}**",
        f"- Readable: **{readable_n}**",
        f"- Unreadable: **{len(rows) - readable_n}**",
        f"- Exact-SHA duplicate members: **{dup_n}**",
        f"- Extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        "",
        "## Mission counts",
        "",
    ]
    for k, v in sorted(mission_c.items()):
        md.append(f"- {k}: {v}")
    md.extend(["", "## Instrument counts", ""])
    for k, v in sorted(instrument_c.items()):
        md.append(f"- {k}: {v}")
    md.append("")
    write_text(out_dir / "source_inventory.md", "\n".join(md))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "iac2026" / "dataset_build",
    )
    args = parser.parse_args(argv)
    root = require_dataset_root()
    print(f"Scanning (read-only): {root}")
    rows = build_inventory(root)
    write_inventory(rows, args.out_dir)
    print(f"Wrote inventory for {len(rows)} files -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
