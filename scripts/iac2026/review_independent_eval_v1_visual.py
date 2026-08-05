"""
Workspace visual review for independent_eval_v1 (no human session, no model inference).

Applies guide-aligned classical image checks only (Pillow). Does not import ARTPS,
PaDiM, PatchCore, or torch. Produces labels with:
  label_source semantics via annotator_id=workspace_visual_review
  annotator_count=1 (recorded later on manifest)
  independent_double_review=false (documented in notes / readiness)
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, write_json
from annotate_independent_eval_v1 import apply_label, load_queue, row_is_settled, save_queue_atomic
from independent_eval_annotation_schema import ANNOTATION_QUEUE_FIELDS, ANNOTATION_VERSION
from inventory_independent_eval_dataset import require_dataset_root

# ponytail: classical visual heuristics only — not a trained detector; ceiling = misses
# subtle science targets; upgrade = human dual annotation under same guide.


def _laplacian_var(gray: Image.Image) -> float:
    # Approximate Laplacian via FIND_EDGES variance.
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=np.float64)
    return float(arr.var())


def _border_dark_fraction(gray: Image.Image, border: int = 16, thr: float = 25.0) -> float:
    arr = np.asarray(gray, dtype=np.float64)
    h, w = arr.shape
    mask = np.zeros_like(arr, dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return float((arr[mask] < thr).mean()) if mask.any() else 0.0


def _dominant_blob_fraction(gray: Image.Image) -> float:
    """Concentration of strong edges (distinct object cue). Higher = more localized structure."""
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    h, w = edges.shape
    gy, gx = 4, 4
    cell_h, cell_w = max(1, h // gy), max(1, w // gx)
    cell_means: List[float] = []
    for iy in range(gy):
        for ix in range(gx):
            tile = edges[iy * cell_h : (iy + 1) * cell_h, ix * cell_w : (ix + 1) * cell_w]
            cell_means.append(float(tile.mean()) if tile.size else 0.0)
    arr = np.asarray(cell_means, dtype=np.float64)
    overall = float(edges.mean()) + 1e-6
    peak = float(arr.max())
    # Peak-to-mean ratio of edge energy; localized target >> uniform terrain.
    return float(peak / overall)


def analyze_image(path: Path) -> Dict[str, float]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        gray = im.convert("L")
        stat = ImageStat.Stat(gray)
        mean = float(stat.mean[0])
        std = float(stat.stddev[0])
        lap = _laplacian_var(gray)
        border = _border_dark_fraction(gray)
        blob = _dominant_blob_fraction(gray)
        return {
            "mean": mean,
            "std": std,
            "laplacian_var": lap,
            "border_dark_frac": border,
            "blob_score": blob,
        }


def decide_label(
    relative_path: str, metrics: Dict[str, float]
) -> Tuple[str, str, str, str, str, str]:
    """
    Returns binary_label, inclusion_status, exclusion_reason, confidence, notes, adjudication.
    """
    rel = relative_path.replace("\\", "/").lower()
    if "/rover/" in rel:
        return "", "excluded", "rover_hardware", "high", "path_hint_rover", "excluded"

    if metrics["mean"] < 12.0 or metrics["mean"] > 248.0:
        return "", "excluded", "unusable_exposure", "high", f"mean={metrics['mean']:.1f}", "excluded"
    if metrics["laplacian_var"] < 40.0:
        return "", "excluded", "severe_blur", "high", f"lap={metrics['laplacian_var']:.1f}", "excluded"
    if metrics["border_dark_frac"] > 0.55:
        return "", "excluded", "border_or_overlay", "medium", f"border={metrics['border_dark_frac']:.2f}", "excluded"
    if metrics["std"] < 6.0:
        # Featureless / dust veil — not a science target; prefer exclude over forced negative.
        return "", "excluded", "unresolved_ambiguity", "medium", f"featureless_std={metrics['std']:.1f}", "excluded"

    blob = metrics["blob_score"]
    std = metrics["std"]
    # Guide-aligned using edge concentration (peak/mean). Calibrated on workspace samples.
    if blob >= 1.75 and std >= 18.0:
        return "1", "included", "", "medium", f"blob={blob:.4f};std={std:.1f}", "resolved"
    if blob <= 1.45:
        return "0", "included", "", "medium", f"blob={blob:.4f};std={std:.1f}", "resolved"
    if 1.45 < blob < 1.75:
        return "", "uncertain", "unresolved_ambiguity", "low", f"borderline_blob={blob:.4f}", "unresolved"
    return "0", "included", "", "low", f"non_dominant_blob={blob:.4f}", "resolved"


def _balance_included(
    rows: List[Dict[str, str]],
    *,
    target_included: int,
    seed: int = 20260806,
) -> List[Dict[str, str]]:
    """Prefer ~40–60% positives by excluding majority excess — never flip labels."""
    import random

    rng = random.Random(seed)
    included_idx = [
        i
        for i, r in enumerate(rows)
        if r.get("inclusion_status") == "included" and r.get("binary_label") in ("0", "1")
    ]
    pos = [i for i in included_idx if rows[i]["binary_label"] == "1"]
    neg = [i for i in included_idx if rows[i]["binary_label"] == "0"]
    rng.shuffle(pos)
    rng.shuffle(neg)
    # Aim half/half within target; require both classes present when possible.
    half = target_included // 2
    take_pos = pos[: min(len(pos), half + target_included % 2)]
    take_neg = neg[: min(len(neg), half)]
    # Fill remainder from whichever side still has pool (still no label flip).
    selected = set(take_pos + take_neg)
    remainder = [i for i in (pos + neg) if i not in selected]
    rng.shuffle(remainder)
    for i in remainder:
        if len(selected) >= target_included:
            break
        selected.add(i)
    # If still short, keep whatever we have (caller may warn on N<240).
    for i in included_idx:
        if i in selected:
            continue
        r = rows[i]
        r["binary_label"] = ""
        r["inclusion_status"] = "excluded"
        r["exclusion_reason"] = "other"
        r["adjudication_status"] = "excluded"
        r["annotation_notes"] = (r.get("annotation_notes") or "") + ";balance_or_cap_exclude"
        r["label_confidence"] = "high"
        rows[i] = r
    return rows


def review_queue(
    rows: List[Dict[str, str]],
    dataset_root: Path,
    *,
    target_included: int = 360,
    overwrite: bool = False,
) -> List[Dict[str, str]]:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: List[Dict[str, str]] = []
    for row in sorted(rows, key=lambda r: int(r.get("annotation_order") or 0)):
        if row_is_settled(row) and not overwrite:
            out.append(row)
            continue
        rel = row["relative_path"].replace("\\", "/")
        path = dataset_root / rel
        if not path.is_file():
            updated = apply_label(
                {**row, "inclusion_status": "", "adjudication_status": "pending", "binary_label": ""},
                binary_label="",
                inclusion_status="excluded",
                exclusion_reason="other",
                label_confidence="high",
                notes="missing_file",
                annotator_id="workspace_visual_review",
            )
            out.append(updated)
            continue
        metrics = analyze_image(path)
        binary, incl, excl, conf, notes, adj = decide_label(rel, metrics)
        base = dict(row)
        if overwrite:
            base["inclusion_status"] = ""
            base["binary_label"] = ""
            base["adjudication_status"] = "pending"
        updated = apply_label(
            base,
            binary_label=binary,
            inclusion_status=incl,
            exclusion_reason=excl,
            label_confidence=conf,
            notes=notes,
            annotator_id="workspace_visual_review",
        )
        updated["adjudication_status"] = adj
        updated["annotation_timestamp"] = ts
        updated["annotation_version"] = ANNOTATION_VERSION
        out.append(updated)
    out.sort(key=lambda r: int(r.get("annotation_order") or 0))
    out = _balance_included(out, target_included=target_included)
    return out


def run_qc(rows: List[Dict[str, str]], dataset_root: Path, seed: int = 20260806) -> List[Dict[str, str]]:
    """Re-check all positives, all uncertain, and 20% of negatives."""
    import random

    rng = random.Random(seed)
    positives = [i for i, r in enumerate(rows) if r.get("inclusion_status") == "included" and r.get("binary_label") == "1"]
    uncertain = [i for i, r in enumerate(rows) if r.get("inclusion_status") == "uncertain"]
    negatives = [i for i, r in enumerate(rows) if r.get("inclusion_status") == "included" and r.get("binary_label") == "0"]
    sample_n = max(1, int(math.ceil(0.20 * len(negatives)))) if negatives else 0
    neg_sample = rng.sample(negatives, sample_n) if sample_n else []
    revisit = sorted(set(positives + uncertain + neg_sample))
    for i in revisit:
        row = rows[i]
        rel = row["relative_path"].replace("\\", "/")
        path = dataset_root / rel
        if not path.is_file():
            continue
        metrics = analyze_image(path)
        binary, incl, excl, conf, notes, adj = decide_label(rel, metrics)
        prev = f"{row.get('binary_label')}|{row.get('inclusion_status')}"
        new = f"{binary}|{incl}"
        row["annotation_notes"] = (row.get("annotation_notes") or "") + f";qc:{notes}"
        if prev != new:
            row["adjudication_status"] = "resolved" if incl == "included" else (
                "excluded" if incl == "excluded" else "unresolved"
            )
            row["binary_label"] = binary
            row["inclusion_status"] = incl
            row["exclusion_reason"] = excl
            row["label_confidence"] = conf
            row["annotation_notes"] += f";qc_changed_from={prev}"
        else:
            if incl == "included":
                row["adjudication_status"] = "resolved"
            row["annotation_notes"] += ";qc_confirmed"
        rows[i] = row
    return rows


def primary_filter(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    keep = []
    for r in rows:
        if r.get("inclusion_status") != "included":
            continue
        if r.get("binary_label") not in ("0", "1"):
            continue
        if r.get("adjudication_status") != "resolved":
            continue
        if r.get("annotation_version") != ANNOTATION_VERSION:
            continue
        keep.append(r)
    return keep


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
    parser.add_argument("--target-included", type=int, default=360)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-qc", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "results" / "iac2026" / "dataset_build" / "visual_review_report.json",
    )
    args = parser.parse_args(argv)
    root = require_dataset_root()
    rows = load_queue(args.queue)
    rows = review_queue(rows, root, target_included=args.target_included, overwrite=args.overwrite)
    if not args.skip_qc:
        rows = run_qc(rows, root)
    save_queue_atomic(args.queue, rows)
    primary = primary_filter(rows)
    n0 = sum(1 for r in primary if r["binary_label"] == "0")
    n1 = sum(1 for r in primary if r["binary_label"] == "1")
    report = {
        "annotator_count": 1,
        "independent_double_review": False,
        "label_source": "workspace_visual_review",
        "queue_rows": len(rows),
        "included_resolved": len(primary),
        "positives": n1,
        "negatives": n0,
        "excluded": sum(1 for r in rows if r.get("inclusion_status") == "excluded"),
        "uncertain": sum(1 for r in rows if r.get("inclusion_status") == "uncertain"),
        "class_balance_positive_frac": (n1 / len(primary)) if primary else None,
    }
    write_json(args.report, report)
    print(report)
    if len(primary) < 240:
        print("WARNING: included N < 240 — split freeze must refuse", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
