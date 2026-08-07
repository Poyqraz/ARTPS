"""
Emit committed, sanitized repeat-author blind-review provenance.

Writes to reproduction/iac2026/annotations/:
  independent_eval_v1_repeat_author_blind_review.csv       (label-only, no PII/paths)
  independent_eval_v1_repeat_author_blind_review.meta.json (SHA provenance)

Does not modify the independent_eval_v1 manifest. Uses no model scores.
Fail-closed unless 54/54 reviews are complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "iac2026"))

from validation_blind_review import (  # noqa: E402
    EXPECTED_VALIDATION_N,
    REVIEW_TYPE_REPEAT_AUTHOR,
    assert_results_complete,
    sha256_file,
    write_sanitized_review_csv,
)

DEFAULT_PACK = (
    REPO_ROOT / "results" / "iac2026" / "independent_eval_v1" / "blind_review_pack"
)
DEFAULT_MANIFEST = (
    REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1.csv"
)
ANNOTATION_GUIDE = (
    REPO_ROOT / "paper" / "iac2026" / "reproduction" / "INDEPENDENT_EVAL_V1_ANNOTATION_GUIDE.md"
)
ANNOTATIONS_DIR = REPO_ROOT / "reproduction" / "iac2026" / "annotations"
OUT_CSV = ANNOTATIONS_DIR / "independent_eval_v1_repeat_author_blind_review.csv"
OUT_META = ANNOTATIONS_DIR / "independent_eval_v1_repeat_author_blind_review.meta.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance best-effort
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = p.parse_args(argv)

    results_path = args.pack_dir / "blind_review_results.csv"
    pack_manifest_path = args.pack_dir / "pack_manifest.json"
    if not results_path.is_file():
        raise SystemExit("missing blind_review_results.csv (run annotator first)")
    results_rows = _read_csv(results_path)
    assert_results_complete(results_rows, n=EXPECTED_VALIDATION_N)

    write_sanitized_review_csv(OUT_CSV, results_rows)

    meta = {
        "review_type": REVIEW_TYPE_REPEAT_AUTHOR,
        "independent_annotator": False,
        "n_reviewed": EXPECTED_VALIDATION_N,
        "review_results_sha256": sha256_file(results_path),
        "blind_pack_manifest_sha256": sha256_file(pack_manifest_path),
        "source_manifest_sha256": sha256_file(args.manifest),
        "annotation_guide_sha256": (
            sha256_file(ANNOTATION_GUIDE) if ANNOTATION_GUIDE.is_file() else None
        ),
        "git_commit": _git_commit(),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_scores_visible": False,
        "original_labels_visible": False,
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV.name} + {OUT_META.name} (n={EXPECTED_VALIDATION_N})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
