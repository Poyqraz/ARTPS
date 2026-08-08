"""
Full 360-sample review comparison for independent_eval_v1_1. FAIL-CLOSED.

Refuses to run unless all 360 samples have genuine human reviews:
  - 54 validation reviews (PR #28 committed sanitized artifact)
  - 306 remaining reviews (train+test) from the local remaining_review_pack

Label-only. Never loads model scores/predictions. Never mutates any manifest.
Emits a gitignored analysis directory only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "iac2026"))

from validation_blind_review import (  # noqa: E402
    ANNOTATION_VERSION_V1_1,
    EXPECTED_REMAINING_N,
    EXPECTED_TOTAL_N,
    EXPECTED_VALIDATION_N,
    FORBIDDEN_COMPARISON_COLUMNS,
    assert_results_complete,
)

DEFAULT_VAL_ARTIFACT = (
    REPO_ROOT
    / "reproduction"
    / "iac2026"
    / "annotations"
    / "independent_eval_v1_repeat_author_blind_review.csv"
)
DEFAULT_VAL_PRIVATE = (
    REPO_ROOT / "results" / "iac2026" / "independent_eval_v1" / "blind_review_pack"
    / "private_mapping.csv"
)
DEFAULT_REMAINING_PACK = (
    REPO_ROOT / "results" / "iac2026" / "independent_eval_v1" / "remaining_review_pack"
)
DEFAULT_MANIFEST = (
    REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1.csv"
)
DEFAULT_OUT = (
    REPO_ROOT / "results" / "iac2026" / "independent_eval_v1" / "v1_1_review_analysis"
)


def _read_csv(path: Path, *, forbid_scores: bool = True) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and forbid_scores:
        for col in rows[0].keys():
            if col in FORBIDDEN_COMPARISON_COLUMNS:
                raise ValueError(f"forbidden comparison column: {col}")
    return rows


def load_full_reviews(
    *,
    val_artifact: Path,
    val_private: Path,
    remaining_pack: Path,
) -> dict[str, dict[str, str]]:
    """Return review_id -> {sample_id, split, reviewer_label, reviewer_confidence}. Fail-closed."""
    if not val_artifact.is_file():
        raise SystemExit(f"pending_review_completion: missing validation artifact {val_artifact}")
    if not val_private.is_file():
        raise SystemExit(
            f"pending_review_completion: missing validation private mapping {val_private}"
        )
    remaining_results = remaining_pack / "blind_review_results.csv"
    remaining_private = remaining_pack / "private_mapping.csv"
    if not remaining_results.is_file():
        raise SystemExit(
            "pending_review_completion: 306 remaining reviews not done "
            f"(missing {remaining_results})"
        )
    if not remaining_private.is_file():
        raise SystemExit(f"pending_review_completion: missing {remaining_private}")

    val_rows = _read_csv(val_artifact)
    if len(val_rows) != EXPECTED_VALIDATION_N:
        raise SystemExit(
            f"pending_review_completion: validation artifact has {len(val_rows)}"
            f" != {EXPECTED_VALIDATION_N}"
        )
    rem_rows = _read_csv(remaining_results)
    try:
        assert_results_complete(rem_rows, n=EXPECTED_REMAINING_N)
    except ValueError as exc:
        raise SystemExit(f"pending_review_completion: {exc}") from exc

    val_priv = {r["review_id"]: r for r in _read_csv(val_private, forbid_scores=False)}
    rem_priv = {r["review_id"]: r for r in _read_csv(remaining_private, forbid_scores=False)}

    combined: dict[str, dict[str, str]] = {}
    for r in val_rows:
        rid = r["review_id"]
        priv = val_priv.get(rid)
        if priv is None:
            raise SystemExit(f"pending_review_completion: no private mapping for {rid}")
        if str(priv.get("split") or "").strip().lower() != "validation":
            raise ValueError(f"validation artifact maps non-validation sample: {rid}")
        combined[rid] = {
            "sample_id": priv["sample_id"],
            "split": priv["split"],
            "reviewer_label": str(r.get("reviewer_label") or "").strip(),
            "reviewer_confidence": str(r.get("reviewer_confidence") or "").strip(),
            "reviewer_role": str(r.get("reviewer_role") or "").strip(),
        }
    for r in rem_rows:
        rid = r["review_id"]
        priv = rem_priv.get(rid)
        if priv is None:
            raise SystemExit(f"pending_review_completion: no private mapping for {rid}")
        split = str(priv.get("split") or "").strip().lower()
        if split not in {"train", "test"}:
            raise ValueError(f"remaining review maps non-train/test sample: {rid} ({split})")
        combined[rid] = {
            "sample_id": priv["sample_id"],
            "split": priv["split"],
            "reviewer_label": str(r.get("reviewer_label") or "").strip(),
            "reviewer_confidence": str(r.get("reviewer_confidence") or "").strip(),
            "reviewer_role": str(r.get("reviewer_role") or "").strip(),
        }

    if len(combined) != EXPECTED_TOTAL_N:
        raise SystemExit(
            f"pending_review_completion: {len(combined)} unique reviews != {EXPECTED_TOTAL_N}"
        )
    return combined


def compare(
    *, reviews: dict[str, dict[str, str]], manifest_rows: list[dict[str, str]]
) -> dict[str, Any]:
    manifest_by_sample = {r["sample_id"]: r for r in manifest_rows}
    agreement = disagreement = uncertain = excluded = 0
    pos_to_neg = neg_to_pos = 0
    conf_dist: Counter[str] = Counter()
    per_split: dict[str, Counter[str]] = {}
    matrix = {
        "original_0_review_0": 0,
        "original_0_review_1": 0,
        "original_1_review_0": 0,
        "original_1_review_1": 0,
    }
    disagreements: list[dict[str, str]] = []

    for rid, rec in reviews.items():
        man = manifest_by_sample.get(rec["sample_id"])
        if man is None:
            raise ValueError(f"sample_id {rec['sample_id']} missing from manifest")
        split = str(man.get("split") or "").strip().lower()
        original = str(man.get("binary_label") or "").strip()
        review = rec["reviewer_label"]
        conf_dist[rec["reviewer_confidence"] or "unset"] += 1
        ps = per_split.setdefault(split, Counter())
        ps["n"] += 1

        if review == "uncertain":
            uncertain += 1
            ps["uncertain"] += 1
            continue
        if review == "exclude":
            excluded += 1
            ps["exclude"] += 1
            continue
        if review not in {"0", "1"} or original not in {"0", "1"}:
            raise ValueError(f"non-binary comparable: orig={original!r} rev={review!r}")
        matrix[f"original_{original}_review_{review}"] += 1
        if review == original:
            agreement += 1
            ps["agree"] += 1
        else:
            disagreement += 1
            ps["disagree"] += 1
            if original == "1":
                pos_to_neg += 1
            else:
                neg_to_pos += 1
            disagreements.append(
                {
                    "review_id": rid,
                    "sample_id": rec["sample_id"],
                    "split": split,
                    "original_binary_label": original,
                    "reviewer_label": review,
                    "reviewer_confidence": rec["reviewer_confidence"],
                }
            )

    comparable = agreement + disagreement
    summary = {
        "comparison_status": "complete",
        "n_reviewed": len(reviews),
        "agreement_count": agreement,
        "disagreement_count": disagreement,
        "uncertain_count": uncertain,
        "excluded_count": excluded,
        "agreement_rate": (agreement / comparable) if comparable else None,
        "disagreement_rate": (disagreement / comparable) if comparable else None,
        "original_positive_to_review_negative": pos_to_neg,
        "original_negative_to_review_positive": neg_to_pos,
        "label_review_confusion_matrix": matrix,
        "confidence_distribution": dict(conf_dist),
        "per_split": {k: dict(v) for k, v in sorted(per_split.items())},
        "model_scores_included": False,
        "manifest_mutated": False,
        "annotation_version_target": ANNOTATION_VERSION_V1_1,
        "review_type": "repeat_author_blind_review",
        "independent_annotator": False,
    }
    return {"summary": summary, "disagreements": disagreements}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-artifact", type=Path, default=DEFAULT_VAL_ARTIFACT)
    p.add_argument("--val-private", type=Path, default=DEFAULT_VAL_PRIVATE)
    p.add_argument("--remaining-pack", type=Path, default=DEFAULT_REMAINING_PACK)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    reviews = load_full_reviews(
        val_artifact=args.val_artifact,
        val_private=args.val_private,
        remaining_pack=args.remaining_pack,
    )
    manifest_rows = _read_csv(args.manifest, forbid_scores=False)
    payload = compare(reviews=reviews, manifest_rows=manifest_rows)
    summary = payload["summary"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "v1_1_review_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "v1_1_review_disagreements.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        fn = [
            "review_id",
            "sample_id",
            "split",
            "original_binary_label",
            "reviewer_label",
            "reviewer_confidence",
        ]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(payload["disagreements"])
    print(json.dumps({"n_reviewed": summary["n_reviewed"], "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
