"""Shared annotation-queue constants for independent_eval_v1 (no inference imports)."""
from __future__ import annotations

ANNOTATION_QUEUE_FIELDS = [
    "candidate_id",
    "relative_path",
    "raw_sha256",
    "mission",
    "instrument",
    "source_id",
    "annotation_order",
    "binary_label",
    "inclusion_status",
    "exclusion_reason",
    "label_confidence",
    "annotation_notes",
    "annotator_id",
    "annotation_timestamp",
    "adjudication_status",
    "annotation_version",
]

FORBIDDEN_QUEUE_COLUMNS = (
    "model_score",
    "anomaly_score",
    "artps_score",
    "heatmap",
    "prediction",
    "padim",
    "patchcore",
)

EXCLUSION_REASONS = (
    "rover_hardware",
    "border_or_overlay",
    "compression_or_sensor_artifact",
    "severe_blur",
    "unusable_exposure",
    "duplicate",
    "unresolved_ambiguity",
    "other",
)

ANNOTATION_VERSION = "independent_eval_v1"
LABEL_SOURCE = "workspace_visual_review"
QUEUE_SEED = 20260806
