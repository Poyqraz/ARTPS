"""Blind-review results schema, normalize, compare fail-closed, freeze guards."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from compare_blind_review_labels import compare, decide  # noqa: E402
from validation_blind_review import (  # noqa: E402
    BLIND_RESULTS_FIELDS,
    EXPECTED_VALIDATION_N,
    FORBIDDEN_VISIBLE_COLUMNS,
    FORBIDDEN_VISIBLE_SUBSTRINGS,
    assert_public_row_blind,
    assert_results_complete,
    build_blind_public_and_private,
    normalize_reviewer_label,
    refuse_mutate_annotation_version,
    results_from_queue_rows,
    write_blind_review_results,
)

MANIFEST = REPO / "reproduction/iac2026/manifests/independent_eval_v1.csv"
FREEZE = REPO / "reproduction/iac2026/test_freeze/FINAL_TEST_SCOPE.yaml"
OPEN = REPO / "reproduction/iac2026/test_freeze/TEST_OPEN_STATUS.yaml"
COMPARE_SCRIPT = REPO / "scripts/iac2026/compare_blind_review_labels.py"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_normalize_positive_negative_identity():
    assert normalize_reviewer_label("positive") == "1"
    assert normalize_reviewer_label("negative") == "0"
    assert normalize_reviewer_label("uncertain") == "uncertain"
    assert normalize_reviewer_label("exclude") == "exclude"
    assert normalize_reviewer_label("0") == "0"
    assert normalize_reviewer_label("1") == "1"
    assert normalize_reviewer_label("") == ""
    with pytest.raises(ValueError):
        normalize_reviewer_label("maybe")


def test_results_from_queue_keeps_raw_and_canonical():
    queue = [
        {
            "review_id": "review_0001",
            "reviewer_label": "positive",
            "reviewer_confidence": "high",
            "reviewer_notes": "x",
        },
        {
            "review_id": "review_0002",
            "reviewer_label": "negative",
            "reviewer_confidence": "low",
            "reviewer_notes": "",
        },
        {"review_id": "review_0003", "reviewer_label": "", "reviewer_confidence": ""},
    ]
    rows = results_from_queue_rows(queue, timestamps={"review_0001": "t1"})
    assert len(rows) == 2
    assert rows[0]["reviewer_label_raw"] == "positive"
    assert rows[0]["reviewer_label"] == "1"
    assert rows[0]["reviewer_decision"] == "1"
    assert rows[1]["reviewer_label_raw"] == "negative"
    assert rows[1]["reviewer_label"] == "0"
    assert list(rows[0].keys()) == BLIND_RESULTS_FIELDS


def test_visible_review_input_has_no_label_score_terrain(tmp_path):
    public, _ = build_blind_public_and_private(_read_csv(MANIFEST))
    for row in public:
        assert_public_row_blind(row)
        for col in FORBIDDEN_VISIBLE_COLUMNS:
            assert col not in row
        blob = " ".join(str(v) for v in row.values()).lower()
        for bad in FORBIDDEN_VISIBLE_SUBSTRINGS:
            assert bad not in blob
        # empty pending labels only — no prior binary labels
        assert row["reviewer_label"] == ""


def test_assert_results_require_54_unique(tmp_path):
    rows = [
        {
            "review_id": f"review_{i:04d}",
            "reviewer_label_raw": "positive" if i % 2 == 0 else "negative",
            "reviewer_label": "1" if i % 2 == 0 else "0",
            "reviewer_confidence": "medium",
            "reviewer_decision": "1" if i % 2 == 0 else "0",
            "reviewer_notes": "",
            "review_timestamp": "t",
            "reviewer_role": "repeat_author_review",
        }
        for i in range(1, 54)
    ]
    with pytest.raises(ValueError, match="54"):
        assert_results_complete(rows)


def test_compare_fail_closed_before_54(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    write_blind_review_results(
        pack / "blind_review_results.csv",
        [
            {
                "review_id": "review_0001",
                "reviewer_label_raw": "positive",
                "reviewer_label": "1",
                "reviewer_confidence": "high",
                "reviewer_decision": "1",
                "reviewer_notes": "",
                "review_timestamp": "t",
                "reviewer_role": "repeat_author_review",
            }
        ],
    )
    (pack / "private_mapping.csv").write_text(
        "review_id,review_order,neutral_filename,sample_id,relative_path,image_sha256,split\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--pack-dir",
            str(pack),
            "--manifest",
            str(MANIFEST),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "pending_review_completion" in (proc.stderr + proc.stdout)


def test_compare_complete_label_only_no_scores(tmp_path):
    public, private = build_blind_public_and_private(_read_csv(MANIFEST))
    man_by = {r["sample_id"]: r for r in _read_csv(MANIFEST)}
    results = []
    for pub, priv in zip(public, private):
        orig = man_by[priv["sample_id"]]["binary_label"]
        raw = "positive" if orig == "1" else "negative"
        results.append(
            {
                "review_id": pub["review_id"],
                "reviewer_label_raw": raw,
                "reviewer_label": orig,
                "reviewer_confidence": "high",
                "reviewer_decision": orig,
                "reviewer_notes": "",
                "review_timestamp": "t",
                "reviewer_role": "repeat_author_review",
            }
        )
    payload = compare(
        results_rows=results,
        private_rows=[{k: str(v) for k, v in p.items()} for p in private],
        manifest_rows=_read_csv(MANIFEST),
    )
    s = payload["summary"]
    assert s["n_reviewed"] == EXPECTED_VALIDATION_N
    assert s["disagreement_count"] == 0
    assert s["decision"] == "labels_confirmed"
    assert s["model_scores_included"] is False
    assert s["manifest_mutated"] is False


def test_decide_thresholds():
    assert (
        decide(
            n_reviewed=54,
            uncertain_count=10,
            excluded_count=5,
            agreement_count=30,
            disagreement_count=9,
        )
        == "excessive_uncertain_or_excluded"
    )
    assert (
        decide(
            n_reviewed=54,
            uncertain_count=0,
            excluded_count=0,
            agreement_count=40,
            disagreement_count=14,
        )
        == "systematic_label_issue_detected"
    )
    assert (
        decide(
            n_reviewed=54,
            uncertain_count=2,
            excluded_count=1,
            agreement_count=48,
            disagreement_count=3,
        )
        == "labels_confirmed"
    )


def test_annotation_change_requires_new_version():
    with pytest.raises(ValueError, match="annotation_version"):
        refuse_mutate_annotation_version(
            current_version="independent_eval_v1",
            requested_version="independent_eval_v1",
        )
    with pytest.raises(ValueError, match="annotation_version"):
        refuse_mutate_annotation_version(
            current_version="independent_eval_v1",
            requested_version=None,
        )
    refuse_mutate_annotation_version(
        current_version="independent_eval_v1",
        requested_version="independent_eval_v1_1",
    )


def test_freeze_still_blocked():
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    open_st = yaml.safe_load(OPEN.read_text(encoding="utf-8"))
    assert freeze["test_opened"] is False
    assert freeze["final_test_authorized"] is False
    assert freeze["status"] == "blocked_validation_sanity_review"
    assert open_st["test_opened"] is False


def test_no_test_artifact_in_blind_private_mapping():
    _, private = build_blind_public_and_private(_read_csv(MANIFEST))
    assert all(str(r["split"]).lower() == "validation" for r in private)
    assert len(private) == 54


def test_report_stub_exists_pending():
    report = REPO / "paper/iac2026/reproduction/INDEPENDENT_EVAL_V1_BLIND_REVIEW_REPORT.md"
    text = report.read_text(encoding="utf-8")
    assert "status: review_pending" in text
    assert "review_type: repeat_author_blind_review" in text
    assert "independent_annotator: false" in text
    assert "reviewed: 0/54" in text
    assert "comparison_status: pending_review_completion" in text
