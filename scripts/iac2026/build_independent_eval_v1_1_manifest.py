"""
Build reproduction/iac2026/manifests/independent_eval_v1_1.csv. FAIL-CLOSED.

Refuses to run until all 360 samples have genuine human reviews (54 validation from
PR #28 + 306 remaining). Writes one provenance row per sample. Never mutates
independent_eval_v1 (byte-for-byte immutability guard). Never auto-binarizes
uncertain/exclude. Never forces class balance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "iac2026"))

from compare_full_review_labels import (  # noqa: E402
    DEFAULT_REMAINING_PACK,
    DEFAULT_VAL_ARTIFACT,
    DEFAULT_VAL_PRIVATE,
    load_full_reviews,
)
from validation_blind_review import (  # noqa: E402
    ANNOTATION_VERSION_V1_1,
    EXPECTED_TOTAL_N,
    REVIEW_SOURCE_REMAINING,
    REVIEW_SOURCE_VALIDATION_PR28,
    V1_1_MANIFEST_FIELDS,
    final_label_from_review,
    review_status_from_label,
)

DEFAULT_MANIFEST_V1 = (
    REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1.csv"
)
DEFAULT_OUT = (
    REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1_1.csv"
)
# Pinned at audit time (INDEPENDENT_EVAL_V1_1_LABEL_AUDIT.md). v1 must stay immutable.
V1_MANIFEST_SHA256 = "1f27e5d74bbf07b47ba8014204328f46a055c31ba4a6f31fb170cf41a910b5fe"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def assert_v1_immutable(manifest_v1: Path) -> str:
    digest = _sha256_file(manifest_v1)
    if digest != V1_MANIFEST_SHA256:
        raise SystemExit(
            "refuse: independent_eval_v1 manifest changed since audit "
            f"(got {digest}, expected {V1_MANIFEST_SHA256}); v1 must stay immutable"
        )
    return digest


def build_rows(
    *, reviews: dict[str, dict[str, str]], manifest_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    by_sample = {r["sample_id"]: r for r in manifest_rows}
    reviewed_samples = {rec["sample_id"] for rec in reviews.values()}
    missing = set(by_sample) - reviewed_samples
    if missing:
        raise SystemExit(
            f"pending_review_completion: {len(missing)} manifest samples unreviewed"
        )

    rows: list[dict[str, str]] = []
    for rec in reviews.values():
        man = by_sample[rec["sample_id"]]
        split = str(man.get("split") or "").strip().lower()
        reviewed = rec["reviewer_label"]
        source = (
            REVIEW_SOURCE_VALIDATION_PR28
            if split == "validation"
            else REVIEW_SOURCE_REMAINING
        )
        rows.append(
            {
                "sample_id": man["sample_id"],
                "split": man.get("split") or "",
                "previous_label": str(man.get("binary_label") or "").strip(),
                "reviewed_label": reviewed,
                "final_label": final_label_from_review(reviewed),
                "review_source": source,
                "reviewer_role": rec["reviewer_role"],
                "reviewer_confidence": rec["reviewer_confidence"],
                "review_status": review_status_from_label(reviewed),
                "annotation_version": ANNOTATION_VERSION_V1_1,
                "scene_group_id": man.get("scene_group_id") or "",
                "duplicate_group_id": man.get("duplicate_group_id") or "",
                "raw_sha256": man.get("raw_sha256") or man.get("sha256") or "",
            }
        )
    rows.sort(key=lambda r: r["sample_id"])
    if len(rows) != EXPECTED_TOTAL_N:
        raise SystemExit(f"expected {EXPECTED_TOTAL_N} rows, got {len(rows)}")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-v1", type=Path, default=DEFAULT_MANIFEST_V1)
    p.add_argument("--val-artifact", type=Path, default=DEFAULT_VAL_ARTIFACT)
    p.add_argument("--val-private", type=Path, default=DEFAULT_VAL_PRIVATE)
    p.add_argument("--remaining-pack", type=Path, default=DEFAULT_REMAINING_PACK)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    v1_sha = assert_v1_immutable(args.manifest_v1)
    reviews = load_full_reviews(
        val_artifact=args.val_artifact,
        val_private=args.val_private,
        remaining_pack=args.remaining_pack,
    )
    manifest_rows = _read_csv(args.manifest_v1)
    rows = build_rows(reviews=reviews, manifest_rows=manifest_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=V1_1_MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    final_dist = Counter(r["final_label"] or "excluded_from_primary" for r in rows)
    status_dist = Counter(r["review_status"] for r in rows)
    meta = {
        "annotation_version": ANNOTATION_VERSION_V1_1,
        "n": len(rows),
        "source_manifest_v1_sha256": v1_sha,
        "final_label_distribution": dict(final_dist),
        "review_status_distribution": dict(status_dist),
        "class_balance_forced": False,
        "auto_binarized_uncertain_exclude": False,
        "manifest_v1_mutated": False,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n": len(rows), "final_label_distribution": dict(final_dist)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
