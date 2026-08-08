"""Emit committed sanitized 360-sample provenance for independent_eval_v1_1.

Label-only (no path/sample/split/score). Fail-closed until 54 PR#28 + 306 remaining
reviews exist. Does not mutate independent_eval_v1.
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

from build_independent_eval_v1_1_manifest import _sha256_file as lf_sha256  # noqa: E402
from validation_blind_review import (  # noqa: E402
    EXPECTED_REMAINING_N,
    EXPECTED_TOTAL_N,
    EXPECTED_VALIDATION_N,
    REVIEW_SOURCE_REMAINING,
    REVIEW_SOURCE_VALIDATION_PR28,
    REVIEW_TYPE_REPEAT_AUTHOR,
    SANITIZED_REVIEW_FIELDS,
    assert_results_complete,
)

ANNOTATIONS_DIR = REPO_ROOT / "reproduction/iac2026/annotations"
VAL_ARTIFACT = ANNOTATIONS_DIR / "independent_eval_v1_repeat_author_blind_review.csv"
REMAINING_RESULTS = (
    REPO_ROOT
    / "results/iac2026/independent_eval_v1/remaining_review_pack/blind_review_results.csv"
)
REMAINING_PACK_MANIFEST = (
    REPO_ROOT / "results/iac2026/independent_eval_v1/remaining_review_pack/pack_manifest.json"
)
V1_MANIFEST = REPO_ROOT / "reproduction/iac2026/manifests/independent_eval_v1.csv"
V11_MANIFEST = REPO_ROOT / "reproduction/iac2026/manifests/independent_eval_v1_1.csv"
GUIDE = REPO_ROOT / "paper/iac2026/reproduction/INDEPENDENT_EVAL_V1_ANNOTATION_GUIDE.md"
OUT_CSV = ANNOTATIONS_DIR / "independent_eval_v1_1_review_provenance.csv"
OUT_META = ANNOTATIONS_DIR / "independent_eval_v1_1_review_provenance.meta.json"
PROVENANCE_FIELDS = SANITIZED_REVIEW_FIELDS + ["review_source"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance best-effort
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    if not VAL_ARTIFACT.is_file():
        raise SystemExit("missing PR#28 sanitized validation artifact")
    if not REMAINING_RESULTS.is_file():
        raise SystemExit("pending_review_completion: missing 306 remaining results")
    if not V11_MANIFEST.is_file():
        raise SystemExit("pending_v1_1_freeze: independent_eval_v1_1.csv missing")

    val_rows = _read_csv(VAL_ARTIFACT)
    rem_rows = _read_csv(REMAINING_RESULTS)
    if len(val_rows) != EXPECTED_VALIDATION_N:
        raise SystemExit(f"validation artifact n={len(val_rows)} != {EXPECTED_VALIDATION_N}")
    assert_results_complete(rem_rows, n=EXPECTED_REMAINING_N)

    out: list[dict[str, str]] = []
    for row in val_rows:
        out.append({k: row.get(k, "") for k in SANITIZED_REVIEW_FIELDS} | {
            "review_source": REVIEW_SOURCE_VALIDATION_PR28
        })
    for row in rem_rows:
        out.append({k: row.get(k, "") for k in SANITIZED_REVIEW_FIELDS} | {
            "review_source": REVIEW_SOURCE_REMAINING
        })
    if len(out) != EXPECTED_TOTAL_N:
        raise SystemExit(f"expected {EXPECTED_TOTAL_N} provenance rows, got {len(out)}")
    ids = [r["review_id"] for r in out]
    if len(set(ids)) != EXPECTED_TOTAL_N:
        raise SystemExit("duplicate review_id in provenance")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDS)
        w.writeheader()
        w.writerows(out)

    meta = {
        "annotation_version": "independent_eval_v1_1",
        "review_type": REVIEW_TYPE_REPEAT_AUTHOR,
        "independent_annotator": False,
        "n_reviewed": EXPECTED_TOTAL_N,
        "n_validation_pr28": EXPECTED_VALIDATION_N,
        "n_remaining": EXPECTED_REMAINING_N,
        "source_manifest_v1_sha256": lf_sha256(V1_MANIFEST),
        "v1_1_manifest_sha256": lf_sha256(V11_MANIFEST),
        "validation_artifact_sha256": lf_sha256(VAL_ARTIFACT),
        "remaining_results_sha256": lf_sha256(REMAINING_RESULTS),
        "remaining_pack_manifest_sha256": (
            lf_sha256(REMAINING_PACK_MANIFEST) if REMAINING_PACK_MANIFEST.is_file() else None
        ),
        "annotation_guide_sha256": lf_sha256(GUIDE) if GUIDE.is_file() else None,
        "git_commit": _git_commit(),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_scores_visible": False,
        "original_labels_visible": False,
        "test_inference_performed": False,
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV.name} + {OUT_META.name} (n={EXPECTED_TOTAL_N})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
