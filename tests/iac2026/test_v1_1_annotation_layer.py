"""independent_eval_v1_1: 360-gate, immutability, blindness, no-auto-binarize guards."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

import build_independent_eval_v1_1_manifest as v11man  # noqa: E402
import build_remaining_review_pack as rempack  # noqa: E402
import compare_full_review_labels as cmp360  # noqa: E402
from validation_blind_review import (  # noqa: E402
    BLIND_QUEUE_FIELDS,
    EXPECTED_REMAINING_N,
    EXPECTED_TOTAL_N,
    EXPECTED_VALIDATION_N,
    FORBIDDEN_VISIBLE_SUBSTRINGS,
    PRIVATE_MAPPING_FIELDS,
    REMAINING_ID_OFFSET,
    assert_public_row_blind,
    build_blind_queue_for_rows,
    final_label_from_review,
    non_validation_rows,
    review_status_from_label,
    validation_rows,
)

MANIFEST = REPO / "reproduction/iac2026/manifests/independent_eval_v1.csv"
FREEZE = REPO / "reproduction/iac2026/test_freeze/FINAL_TEST_SCOPE.yaml"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_360_accounted_54_val_306_nonval():
    rows = _read_csv(MANIFEST)
    val = validation_rows(rows)
    nonval = non_validation_rows(rows)
    assert len(val) == EXPECTED_VALIDATION_N == 54
    assert len(nonval) == EXPECTED_REMAINING_N == 306
    assert len(val) + len(nonval) == EXPECTED_TOTAL_N == 360


def test_remaining_pack_continuation_ids_and_blind():
    rows = non_validation_rows(_read_csv(MANIFEST))
    public, private = build_blind_queue_for_rows(
        rows, id_offset=REMAINING_ID_OFFSET
    )
    assert len(public) == 306
    ids = [r["review_id"] for r in public]
    assert ids[0] == "review_0055" and ids[-1] == "review_0360"
    assert len(set(ids)) == 306
    # public queue is model-blind and split-hidden
    assert "split" not in BLIND_QUEUE_FIELDS
    assert "split" in PRIVATE_MAPPING_FIELDS
    for row in public:
        assert_public_row_blind(row)
        blob = " ".join(str(v).lower() for v in row.values())
        for banned in FORBIDDEN_VISIBLE_SUBSTRINGS:
            assert banned not in blob


def test_private_mapping_carries_test_split_but_public_does_not():
    rows = non_validation_rows(_read_csv(MANIFEST))
    _public, private = build_blind_queue_for_rows(rows, id_offset=REMAINING_ID_OFFSET)
    splits = {str(p["split"]).strip().lower() for p in private}
    assert "test" in splits  # test images ARE annotated by humans
    assert splits <= {"train", "test"}


def test_no_auto_binarize_uncertain_exclude():
    assert final_label_from_review("uncertain") == ""
    assert final_label_from_review("exclude") == ""
    assert final_label_from_review("1") == "1"
    assert final_label_from_review("0") == "0"
    assert review_status_from_label("uncertain") == "reviewed_uncertain"
    assert review_status_from_label("exclude") == "reviewed_exclude"
    assert review_status_from_label("1") == "reviewed"


def test_heuristic_label_not_promoted_to_reviewed():
    # An empty/absent review must never become a binary "reviewed" label.
    assert final_label_from_review("") == ""
    assert review_status_from_label("") == "pending_manual_review"


def test_v1_manifest_immutable_sha_pinned():
    import hashlib

    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert digest == v11man.V1_MANIFEST_SHA256
    # guard raises if v1 ever changes
    v11man.assert_v1_immutable(MANIFEST)


def test_compare_360_fail_closed_before_remaining(tmp_path):
    with pytest.raises(SystemExit):
        cmp360.load_full_reviews(
            val_artifact=cmp360.DEFAULT_VAL_ARTIFACT,
            val_private=cmp360.DEFAULT_VAL_PRIVATE,
            remaining_pack=tmp_path,  # empty -> no 306 results
        )


def test_v1_1_manifest_builder_fail_closed_before_360(tmp_path):
    with pytest.raises(SystemExit):
        v11man.main(
            [
                "--remaining-pack",
                str(tmp_path),
                "--out",
                str(tmp_path / "independent_eval_v1_1.csv"),
            ]
        )
    assert not (tmp_path / "independent_eval_v1_1.csv").exists()


def test_no_forced_balance_or_inference_in_tooling():
    man_src = (REPO / "scripts/iac2026/build_independent_eval_v1_1_manifest.py").read_text(
        encoding="utf-8"
    )
    assert "_balance_included" not in man_src and "target_included" not in man_src
    pack_src = (REPO / "scripts/iac2026/build_remaining_review_pack.py").read_text(
        encoding="utf-8"
    )
    # No actual inference imports/calls (docstring may mention them as forbidden).
    code_lines = [
        ln
        for ln in pack_src.splitlines()
        if ln.strip().startswith(("import ", "from "))
    ]
    code = "\n".join(code_lines)
    for banned in ("artps_inference", "PaDiM", "PatchCore"):
        assert banned not in code
    assert ".predict(" not in pack_src


def test_test_split_embargo_intact():
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    assert freeze["test_opened"] is False
    assert freeze["final_test_authorized"] is False
