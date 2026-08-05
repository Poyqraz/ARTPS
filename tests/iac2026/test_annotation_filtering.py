"""Primary-eval annotation filtering rules."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from review_independent_eval_v1_visual import primary_filter  # noqa: E402


def test_unresolved_annotation_excluded():
    rows = [
        {
            "inclusion_status": "included",
            "binary_label": "1",
            "adjudication_status": "unresolved",
            "annotation_version": "independent_eval_v1",
        },
        {
            "inclusion_status": "included",
            "binary_label": "0",
            "adjudication_status": "resolved",
            "annotation_version": "independent_eval_v1",
        },
    ]
    keep = primary_filter(rows)
    assert len(keep) == 1
    assert keep[0]["binary_label"] == "0"


def test_invalid_label_rejected():
    rows = [
        {
            "inclusion_status": "included",
            "binary_label": "2",
            "adjudication_status": "resolved",
            "annotation_version": "independent_eval_v1",
        }
    ]
    assert primary_filter(rows) == []


def test_included_requires_resolved():
    rows = [
        {
            "inclusion_status": "included",
            "binary_label": "1",
            "adjudication_status": "pending",
            "annotation_version": "independent_eval_v1",
        }
    ]
    assert primary_filter(rows) == []
