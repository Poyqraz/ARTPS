"""Independent manifest build constraints."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from build_independent_eval_manifest import select_primary_rows  # noqa: E402
from build_independent_eval_split import SplitContractError, assert_no_aggregate_quota  # noqa: E402


def test_aggregate_historical_counts_cannot_create_rows():
    with pytest.raises(SplitContractError):
        assert_no_aggregate_quota("expanded from aggregate 2847 quota")


def test_manifest_primary_filter():
    rows = [
        {
            "candidate_id": "ok",
            "inclusion_status": "included",
            "binary_label": "1",
            "adjudication_status": "resolved",
            "annotation_version": "independent_eval_v1",
            "annotation_notes": "",
            "raw_sha256": "a" * 64,
            "relative_path": "train/x.jpg",
        },
        {
            "candidate_id": "bad",
            "inclusion_status": "uncertain",
            "binary_label": "",
            "adjudication_status": "unresolved",
            "annotation_version": "independent_eval_v1",
            "annotation_notes": "",
            "raw_sha256": "b" * 64,
            "relative_path": "train/y.jpg",
        },
    ]
    keep = select_primary_rows(rows)
    assert [r["candidate_id"] for r in keep] == ["ok"]
