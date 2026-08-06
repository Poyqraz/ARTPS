"""Blind validation review queue schema (no labels/paths/scores in public view)."""
from __future__ import annotations

from typing import Any, Iterable

BLIND_QUEUE_SEED = 20260806

BLIND_QUEUE_FIELDS = [
    "review_order",
    "review_id",
    "neutral_filename",
    "image_sha256",
    "audit_status",
    "seed",
    "reviewer_label",
    "reviewer_confidence",
    "reviewer_notes",
]

PRIVATE_MAPPING_FIELDS = [
    "review_id",
    "review_order",
    "neutral_filename",
    "sample_id",
    "relative_path",
    "image_sha256",
    "split",
]

FORBIDDEN_VISIBLE_SUBSTRINGS = (
    "train",
    "valid",
    "test",
    "rocky",
    "dusty",
    "boulder",
    "flat_terrain",
    "hills_or_ridge",
)

FORBIDDEN_VISIBLE_COLUMNS = (
    "sample_id",
    "relative_path",
    "path_exists",
    "binary_label",
    "y_true",
    "anomaly_score",
    "image_score",
    "split",
    "candidate_count",
)

UNAVAILABLE_SUPPRESSION = "unavailable_requires_instrumented_validation_rerun"

DECISION_TEXT_SCOPED = (
    "No metric-computation or manifest-to-prediction label-mapping defect was "
    "detected in the committed artifacts. Score-component semantics, classifier "
    "class ordering, candidate suppression behavior, and independent annotation "
    "quality remain unresolved."
)


def is_included_resolved(row: dict[str, str]) -> bool:
    incl = str(row.get("inclusion_status") or row.get("included") or "").strip().lower()
    adj = str(row.get("adjudication_status") or row.get("resolved") or "").strip().lower()
    return incl in {"included", "1", "true", "yes"} and adj in {
        "resolved",
        "1",
        "true",
        "yes",
    }


def validation_rows(manifest_rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [
        r
        for r in manifest_rows
        if str(r.get("split")).strip().lower() == "validation" and is_included_resolved(r)
    ]


def build_blind_public_and_private(
    manifest_rows: list[dict[str, str]],
    *,
    seed: int = BLIND_QUEUE_SEED,
    permutation: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic blind queue + private mapping. permutation overrides RNG when provided."""
    import numpy as np

    val = validation_rows(manifest_rows)
    if permutation is None:
        rng = np.random.default_rng(seed)
        order = [int(i) for i in rng.permutation(len(val))]
    else:
        order = list(permutation)
        if sorted(order) != list(range(len(val))):
            raise ValueError("permutation must be a permutation of range(n)")

    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for i, idx in enumerate(order):
        r = val[idx]
        review_id = f"review_{i + 1:04d}"
        neutral = f"{review_id}.jpg"
        sha = (r.get("sha256") or r.get("raw_sha256") or "").strip()
        public.append(
            {
                "review_order": i,
                "review_id": review_id,
                "neutral_filename": neutral,
                "image_sha256": sha,
                "audit_status": "pending_independent_review",
                "seed": seed,
                "reviewer_label": "",
                "reviewer_confidence": "",
                "reviewer_notes": "",
            }
        )
        private.append(
            {
                "review_id": review_id,
                "review_order": i,
                "neutral_filename": neutral,
                "sample_id": r["sample_id"],
                "relative_path": r.get("relative_path") or "",
                "image_sha256": sha,
                "split": r.get("split") or "",
            }
        )
    return public, private


def assert_public_row_blind(row: dict[str, Any]) -> None:
    blob = " ".join(str(v) for v in row.values()).lower()
    for bad in FORBIDDEN_VISIBLE_SUBSTRINGS:
        if bad in blob:
            raise ValueError(f"forbidden substring {bad!r} in public blind row")
    for col in FORBIDDEN_VISIBLE_COLUMNS:
        if col in row:
            raise ValueError(f"forbidden column {col!r} in public blind row")
