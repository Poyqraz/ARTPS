"""
Compare completed blind-review labels to original manifest binary_label.

Fail-closed until 54 unique completed reviews exist.
Does not load model scores or predictions.
Does not mutate the independent_eval_v1 manifest.

Decision thresholds (label-only):
  C excessive_uncertain_or_excluded:
      (uncertain + exclude) / n_reviewed >= 0.25
  B systematic_label_issue_detected:
      among pairs where both original and review are in {0,1},
      disagreement_rate >= 0.20
  A labels_confirmed: otherwise

Outputs (local / gitignored analysis dir):
  blind_review_summary.json
  blind_review_disagreements.csv
  blind_review_summary.md
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
    EXPECTED_VALIDATION_N,
    EXCESSIVE_UNCERTAIN_OR_EXCLUDE_RATE,
    FORBIDDEN_COMPARISON_COLUMNS,
    SYSTEMATIC_DISAGREEMENT_RATE,
    assert_results_complete,
    refuse_mutate_annotation_version,
)

DEFAULT_PACK = (
    REPO_ROOT / "results" / "iac2026" / "independent_eval_v1" / "blind_review_pack"
)
DEFAULT_MANIFEST = (
    REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1.csv"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "results"
    / "iac2026"
    / "independent_eval_v1"
    / "blind_review_analysis"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows:
        for col in rows[0].keys():
            if col in FORBIDDEN_COMPARISON_COLUMNS:
                raise ValueError(f"forbidden comparison column: {col}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def decide(
    *,
    n_reviewed: int,
    uncertain_count: int,
    excluded_count: int,
    agreement_count: int,
    disagreement_count: int,
) -> str:
    if n_reviewed <= 0:
        return "pending_review_completion"
    ue_rate = (uncertain_count + excluded_count) / n_reviewed
    if ue_rate >= EXCESSIVE_UNCERTAIN_OR_EXCLUDE_RATE:
        return "excessive_uncertain_or_excluded"
    comparable = agreement_count + disagreement_count
    if comparable > 0 and (disagreement_count / comparable) >= SYSTEMATIC_DISAGREEMENT_RATE:
        return "systematic_label_issue_detected"
    return "labels_confirmed"


def compare(
    *,
    results_rows: list[dict[str, str]],
    private_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
) -> dict[str, Any]:
    assert_results_complete(results_rows, n=EXPECTED_VALIDATION_N)

    private_by_id = {r["review_id"]: r for r in private_rows}
    if len(private_by_id) != EXPECTED_VALIDATION_N:
        raise ValueError(
            f"private_mapping expected {EXPECTED_VALIDATION_N}, got {len(private_by_id)}"
        )
    for rid in private_by_id:
        if str(private_by_id[rid].get("split") or "").lower() == "test":
            raise ValueError("test split leaked into private mapping")

    manifest_by_sample = {r["sample_id"]: r for r in manifest_rows}

    disagreements: list[dict[str, str]] = []
    agreement_count = 0
    disagreement_count = 0
    uncertain_count = 0
    excluded_count = 0
    pos_to_neg = 0
    neg_to_pos = 0
    original_positive_count = 0
    original_negative_count = 0
    reviewed_positive_count = 0
    reviewed_negative_count = 0
    # label_review_confusion_matrix (NOT a model prediction confusion matrix)
    matrix = {
        "original_0_review_0": 0,
        "original_0_review_1": 0,
        "original_1_review_0": 0,
        "original_1_review_1": 0,
    }
    conf_dist: Counter[str] = Counter()

    for row in results_rows:
        rid = row["review_id"]
        priv = private_by_id[rid]
        sample_id = priv["sample_id"]
        man = manifest_by_sample.get(sample_id)
        if man is None:
            raise ValueError(f"sample_id {sample_id} missing from manifest")
        if str(man.get("split") or "").lower() != "validation":
            raise ValueError(f"non-validation sample in comparison: {sample_id}")

        original = str(man.get("binary_label") or "").strip()
        review = str(row.get("reviewer_label") or "").strip()
        conf = str(row.get("reviewer_confidence") or "").strip() or "unset"
        conf_dist[conf] += 1

        if review == "uncertain":
            uncertain_count += 1
            continue
        if review == "exclude":
            excluded_count += 1
            continue
        if review not in {"0", "1"} or original not in {"0", "1"}:
            raise ValueError(f"non-binary comparable labels: orig={original!r} rev={review!r}")

        matrix[f"original_{original}_review_{review}"] += 1
        if original == "1":
            original_positive_count += 1
        else:
            original_negative_count += 1
        if review == "1":
            reviewed_positive_count += 1
        else:
            reviewed_negative_count += 1

        if review == original:
            agreement_count += 1
        else:
            disagreement_count += 1
            if original == "1" and review == "0":
                pos_to_neg += 1
            elif original == "0" and review == "1":
                neg_to_pos += 1
            disagreements.append(
                {
                    "review_id": rid,
                    "sample_id": sample_id,
                    "original_binary_label": original,
                    "reviewer_label": review,
                    "reviewer_label_raw": row.get("reviewer_label_raw") or "",
                    "reviewer_confidence": row.get("reviewer_confidence") or "",
                }
            )

    n_reviewed = len(results_rows)
    comparable = agreement_count + disagreement_count
    agreement_rate = (agreement_count / comparable) if comparable else None
    disagreement_rate = (disagreement_count / comparable) if comparable else None
    decision = decide(
        n_reviewed=n_reviewed,
        uncertain_count=uncertain_count,
        excluded_count=excluded_count,
        agreement_count=agreement_count,
        disagreement_count=disagreement_count,
    )

    summary: dict[str, Any] = {
        "comparison_status": "complete",
        "n_reviewed": n_reviewed,
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "uncertain_count": uncertain_count,
        "excluded_count": excluded_count,
        "agreement_rate": agreement_rate,
        "disagreement_rate": disagreement_rate,
        "original_positive_count": original_positive_count,
        "original_negative_count": original_negative_count,
        "reviewed_positive_count": reviewed_positive_count,
        "reviewed_negative_count": reviewed_negative_count,
        "label_review_confusion_matrix": matrix,
        "original_positive_to_review_negative": pos_to_neg,
        "original_negative_to_review_positive": neg_to_pos,
        "confidence_distribution": dict(conf_dist),
        "decision": decision,
        "thresholds": {
            "excessive_uncertain_or_exclude_rate": EXCESSIVE_UNCERTAIN_OR_EXCLUDE_RATE,
            "systematic_disagreement_rate": SYSTEMATIC_DISAGREEMENT_RATE,
        },
        "manifest_mutated": False,
        "annotation_version_changed": False,
        "suggested_annotation_version_if_B": ANNOTATION_VERSION_V1_1,
        "model_scores_included": False,
        "review_type": "repeat_author_blind_review",
        "independent_annotator": False,
    }
    return {"summary": summary, "disagreements": disagreements}


def render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Blind review comparison summary",
        "",
        f"- comparison_status: `{summary['comparison_status']}`",
        f"- review_type: `{summary['review_type']}`",
        f"- independent_annotator: `{summary['independent_annotator']}`",
        f"- n_reviewed: {summary['n_reviewed']}",
        f"- agreement_count: {summary['agreement_count']}",
        f"- disagreement_count: {summary['disagreement_count']}",
        f"- uncertain_count: {summary['uncertain_count']}",
        f"- excluded_count: {summary['excluded_count']}",
        f"- agreement_rate: {summary['agreement_rate']}",
        f"- disagreement_rate: {summary['disagreement_rate']}",
        f"- original_positive_count: {summary['original_positive_count']}",
        f"- original_negative_count: {summary['original_negative_count']}",
        f"- reviewed_positive_count: {summary['reviewed_positive_count']}",
        f"- reviewed_negative_count: {summary['reviewed_negative_count']}",
        f"- original_positive_to_review_negative: {summary['original_positive_to_review_negative']}",
        f"- original_negative_to_review_positive: {summary['original_negative_to_review_positive']}",
        f"- confidence_distribution: `{summary['confidence_distribution']}`",
        f"- decision: `{summary['decision']}`",
        "",
        "## label_review_confusion_matrix (NOT a model prediction confusion matrix)",
        "",
        "| original \\ review | review 0 | review 1 |",
        "|---|---|---|",
        f"| original 0 | {summary['label_review_confusion_matrix']['original_0_review_0']} | {summary['label_review_confusion_matrix']['original_0_review_1']} |",
        f"| original 1 | {summary['label_review_confusion_matrix']['original_1_review_0']} | {summary['label_review_confusion_matrix']['original_1_review_1']} |",
        "",
        "Manifest was not modified. Model scores were not included.",
        "",
    ]
    if summary["decision"] == "systematic_label_issue_detected":
        lines.append(
            f"Suggested next annotation_version (do not auto-apply): `{summary['suggested_annotation_version_if_B']}`"
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--propose-annotation-version",
        default=None,
        help="If set equal to current version, refuse (immutability guard).",
    )
    args = p.parse_args(argv)

    results_path = args.pack_dir / "blind_review_results.csv"
    private_path = args.pack_dir / "private_mapping.csv"
    if not results_path.is_file():
        raise SystemExit(
            "comparison_status=pending_review_completion: missing blind_review_results.csv"
        )
    results_rows = _read_csv(results_path)
    try:
        assert_results_complete(results_rows, n=EXPECTED_VALIDATION_N)
    except ValueError as exc:
        raise SystemExit(f"comparison_status=pending_review_completion: {exc}") from exc

    if args.propose_annotation_version is not None:
        # Guard demo: mutating v1 in place is forbidden.
        refuse_mutate_annotation_version(
            current_version="independent_eval_v1",
            requested_version=args.propose_annotation_version,
        )

    private_rows = _read_csv(private_path)
    manifest_rows = _read_csv(args.manifest)
    payload = compare(
        results_rows=results_rows,
        private_rows=private_rows,
        manifest_rows=manifest_rows,
    )
    summary = payload["summary"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "blind_review_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        args.out_dir / "blind_review_disagreements.csv",
        payload["disagreements"],
        [
            "review_id",
            "sample_id",
            "original_binary_label",
            "reviewer_label",
            "reviewer_label_raw",
            "reviewer_confidence",
        ],
    )
    (args.out_dir / "blind_review_summary.md").write_text(
        render_md(summary), encoding="utf-8"
    )
    print(json.dumps({"decision": summary["decision"], "n_reviewed": summary["n_reviewed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
