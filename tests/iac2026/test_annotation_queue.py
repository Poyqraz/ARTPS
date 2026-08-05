"""Annotation queue schema and model-score column bans."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from build_independent_eval_annotation_queue import (  # noqa: E402
    assert_no_forbidden_columns,
    build_queue,
)
from independent_eval_annotation_schema import (  # noqa: E402
    ANNOTATION_QUEUE_FIELDS,
    FORBIDDEN_QUEUE_COLUMNS,
)


def test_queue_fields_forbid_model_score_columns():
    assert_no_forbidden_columns(ANNOTATION_QUEUE_FIELDS)
    with pytest.raises(ValueError, match="forbidden"):
        assert_no_forbidden_columns(list(ANNOTATION_QUEUE_FIELDS) + ["model_score"])


def test_forbidden_column_names_listed():
    assert "model_score" in FORBIDDEN_QUEUE_COLUMNS
    assert "heatmap" in FORBIDDEN_QUEUE_COLUMNS


def test_build_queue_deterministic_order():
    inv = [
        {
            "candidate_id": f"c{i}",
            "relative_path": f"train/rocky/curiosity_100_MAST_{i}.jpg",
            "filename": f"curiosity_100_MAST_{i}.jpg",
            "raw_sha256": f"{i:064d}"[:64].replace(" ", "0"),
            "mission": "Curiosity",
            "instrument": "Mastcam",
            "source_id": "UNKNOWN",
            "readable": "true",
        }
        for i in range(5)
    ]
    # pad sha
    for r in inv:
        r["raw_sha256"] = (r["raw_sha256"] + "0" * 64)[:64]
    a = build_queue(inv, seed=7)
    b = build_queue(inv, seed=7)
    assert [r["candidate_id"] for r in a] == [r["candidate_id"] for r in b]
    assert {r["annotation_order"] for r in a} == {str(i) for i in range(5)}
