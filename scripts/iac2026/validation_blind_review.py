"""Blind validation review queue schema (no labels/paths/scores in public view)."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

BLIND_QUEUE_SEED = 20260806
EXPECTED_VALIDATION_N = 54

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

# Canonical export (raw preserved separately; never overwrite queue raw in place).
BLIND_RESULTS_FIELDS = [
    "review_id",
    "reviewer_label_raw",
    "reviewer_label",
    "reviewer_confidence",
    "reviewer_decision",
    "reviewer_notes",
    "review_timestamp",
    "reviewer_role",
]

CANONICAL_LABELS = frozenset({"0", "1", "uncertain", "exclude"})
RAW_UI_LABELS = frozenset({"positive", "negative", "uncertain", "exclude", "0", "1"})

LABEL_NORMALIZE = {
    "positive": "1",
    "negative": "0",
    "uncertain": "uncertain",
    "exclude": "exclude",
    "0": "0",
    "1": "1",
}

REVIEWER_ROLE_REPEAT_AUTHOR = "repeat_author_review"
REVIEW_TYPE_REPEAT_AUTHOR = "repeat_author_blind_review"

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

# Committed sanitized provenance: no path/sample/original-label/model data.
SANITIZED_REVIEW_FIELDS = [
    "review_id",
    "reviewer_label",
    "reviewer_confidence",
    "reviewer_decision",
    "reviewer_role",
]

# Comparison must never ingest model outputs.
FORBIDDEN_COMPARISON_COLUMNS = (
    "anomaly_score",
    "image_score",
    "prediction",
    "y_pred",
    "score",
    "candidate_count",
)

UNAVAILABLE_SUPPRESSION = "unavailable_requires_instrumented_validation_rerun"

DECISION_TEXT_SCOPED = (
    "No metric-computation or manifest-to-prediction label-mapping defect was "
    "detected in the committed artifacts. Score-component semantics, classifier "
    "class ordering, candidate suppression behavior, and independent annotation "
    "quality remain unresolved."
)

ANNOTATION_VERSION_V1 = "independent_eval_v1"
ANNOTATION_VERSION_V1_1 = "independent_eval_v1_1"

# Decision thresholds for compare_blind_review_labels (label-only; no scores).
EXCESSIVE_UNCERTAIN_OR_EXCLUDE_RATE = 0.25
SYSTEMATIC_DISAGREEMENT_RATE = 0.20


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


def normalize_reviewer_label(raw: str | None) -> str:
    """Map UI/raw label to canonical reviewer_label. Empty raw stays empty."""
    text = str(raw or "").strip()
    if not text:
        return ""
    key = text.lower()
    if key not in LABEL_NORMALIZE:
        raise ValueError(f"unknown reviewer_label_raw: {raw!r}")
    return LABEL_NORMALIZE[key]


def build_results_row(
    *,
    review_id: str,
    reviewer_label_raw: str,
    reviewer_confidence: str,
    reviewer_notes: str,
    review_timestamp: str,
    reviewer_role: str = REVIEWER_ROLE_REPEAT_AUTHOR,
) -> dict[str, str]:
    canonical = normalize_reviewer_label(reviewer_label_raw)
    return {
        "review_id": review_id,
        "reviewer_label_raw": reviewer_label_raw,
        "reviewer_label": canonical,
        "reviewer_confidence": reviewer_confidence,
        "reviewer_decision": canonical,
        "reviewer_notes": reviewer_notes,
        "review_timestamp": review_timestamp,
        "reviewer_role": reviewer_role,
    }


def write_blind_review_results(path: Path, rows: list[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BLIND_RESULTS_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in BLIND_RESULTS_FIELDS})
    tmp.replace(path)


def results_from_queue_rows(
    queue_rows: Iterable[Mapping[str, Any]],
    *,
    timestamps: Mapping[str, str] | None = None,
    reviewer_role: str = REVIEWER_ROLE_REPEAT_AUTHOR,
) -> list[dict[str, str]]:
    """Build canonical results for rows that have a non-empty raw label."""
    out: list[dict[str, str]] = []
    ts_map = timestamps or {}
    for row in queue_rows:
        raw = str(row.get("reviewer_label") or "").strip()
        if not raw:
            continue
        rid = str(row["review_id"])
        out.append(
            build_results_row(
                review_id=rid,
                reviewer_label_raw=raw,
                reviewer_confidence=str(row.get("reviewer_confidence") or ""),
                reviewer_notes=str(row.get("reviewer_notes") or ""),
                review_timestamp=str(ts_map.get(rid) or row.get("review_timestamp") or ""),
                reviewer_role=reviewer_role,
            )
        )
    return out


def assert_results_complete(rows: list[Mapping[str, Any]], *, n: int = EXPECTED_VALIDATION_N) -> None:
    ids = [str(r.get("review_id") or "") for r in rows]
    if len(ids) != n:
        raise ValueError(f"expected {n} review results, got {len(ids)}")
    if len(set(ids)) != n:
        raise ValueError("review_id values must be unique")
    for row in rows:
        raw = str(row.get("reviewer_label_raw") or "").strip()
        canon = str(row.get("reviewer_label") or "").strip()
        if not raw:
            raise ValueError(f"empty reviewer_label_raw for {row.get('review_id')}")
        if canon not in CANONICAL_LABELS:
            raise ValueError(f"invalid canonical label {canon!r} for {row.get('review_id')}")
        if normalize_reviewer_label(raw) != canon:
            raise ValueError(
                f"raw/canonical mismatch for {row.get('review_id')}: {raw!r} vs {canon!r}"
            )


def refuse_mutate_annotation_version(
    *,
    current_version: str,
    requested_version: str | None,
) -> None:
    """Label edits require a new annotation_version; old manifest stays immutable."""
    cur = (current_version or "").strip()
    req = (requested_version or "").strip() if requested_version is not None else ""
    if not req or req == cur:
        raise ValueError(
            "annotation change requires new annotation_version "
            f"(current={cur!r}; proposed={ANNOTATION_VERSION_V1_1!r})"
        )


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_sanitized_review_rows(results_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Committed provenance rows: only review_id/label/confidence/decision/role."""
    out: list[dict[str, str]] = []
    for row in results_rows:
        sanitized = {k: str(row.get(k, "")) for k in SANITIZED_REVIEW_FIELDS}
        for banned in ("sample_id", "relative_path", "neutral_filename", "image_sha256"):
            if banned in row:
                raise ValueError(f"sanitized artifact must not carry {banned!r}")
        out.append(sanitized)
    return out


def write_sanitized_review_csv(path: Path, results_rows: Iterable[Mapping[str, Any]]) -> None:
    rows = build_sanitized_review_rows(results_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SANITIZED_REVIEW_FIELDS)
        w.writeheader()
        w.writerows(rows)


def repeat_author_review_meta(**extra: Any) -> dict[str, Any]:
    meta = {
        "review_type": REVIEW_TYPE_REPEAT_AUTHOR,
        "independent_annotator": False,
        "independent_review_status": "pending",
        "model_blind": True,
        "existing_label_hidden": True,
        "terrain_hidden": True,
        "source_partition_hidden": True,
        "comparison_status": "pending_review_completion",
        "repeat_author_review": True,
        "note": "Not an independent second annotation. private_mapping.csv must never load in UI.",
    }
    meta.update(extra)
    return meta
