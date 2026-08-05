"""Split ratio policy depends only on included N."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from build_independent_eval_split import (  # noqa: E402
    SplitContractError,
    assign_splits,
    check_group_leakage,
    refuse_frozen_test_mutation,
    select_split_ratios,
)


def test_split_ratio_selected_only_by_n():
    assert select_split_ratios(400) == (0.70, 0.15, 0.15)
    assert select_split_ratios(360) == (0.70, 0.15, 0.15)
    assert select_split_ratios(300) == (0.60, 0.20, 0.20)
    assert select_split_ratios(240) == (0.60, 0.20, 0.20)
    assert select_split_ratios(239) is None


def test_minimum_n_policy():
    assert select_split_ratios(0) is None


def test_cross_split_groups_rejected():
    rows = [
        {"sha256": "aa", "duplicate_group_id": "d1", "scene_group_id": "s1", "source_id": "x", "sequence_id": "1", "split": "train"},
        {"sha256": "bb", "duplicate_group_id": "d1", "scene_group_id": "s2", "source_id": "y", "sequence_id": "2", "split": "test"},
    ]
    errs = check_group_leakage(rows, ["duplicate_group_id"])
    assert errs


def test_frozen_test_mutation_rejected(tmp_path):
    marker = tmp_path / "TEST_SPLIT_FROZEN"
    out = tmp_path / "out.csv"
    marker.write_text("{}", encoding="utf-8")
    out.write_text("x\n", encoding="utf-8")
    with pytest.raises(SplitContractError, match="frozen"):
        refuse_frozen_test_mutation(freeze_marker=marker, output_manifest=out)


def test_assign_splits_both_classes_in_val_test():
    rows = []
    for i in range(120):
        rows.append(
            {
                "sample_id": f"p{i}",
                "binary_label": "1",
                "duplicate_group_id": f"dp{i}",
                "sha256": f"{i:064d}"[:64].ljust(64, "0"),
                "inclusion_status": "included",
                "adjudication_status": "resolved",
            }
        )
    for i in range(120):
        rows.append(
            {
                "sample_id": f"n{i}",
                "binary_label": "0",
                "duplicate_group_id": f"dn{i}",
                "sha256": f"{i+500:064d}"[:64].ljust(64, "0"),
                "inclusion_status": "included",
                "adjudication_status": "resolved",
            }
        )
    assigned = assign_splits(rows, (0.60, 0.20, 0.20), seed=1)
    from collections import Counter, defaultdict

    c = defaultdict(Counter)
    for r in assigned:
        c[r["split"]][r["binary_label"]] += 1
    assert c["validation"]["0"] > 0 and c["validation"]["1"] > 0
    assert c["test"]["0"] > 0 and c["test"]["1"] > 0
