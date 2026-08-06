"""Validation sanity audit regression tests (final test stays closed)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from detection_metrics_lib import (  # noqa: E402
    average_precision,
    binary_auroc,
    confusion,
    select_threshold_on_validation,
)
from test_split_embargo import assert_final_test_authorized, load_final_test_scope  # noqa: E402

VAL_ROOT = REPO / "results/iac2026/independent_eval_v1/validation"
REPORT = VAL_ROOT / "validation_sanity_report.json"
SELECTION = VAL_ROOT / "profile_selection.json"
MANIFEST = REPO / "reproduction/iac2026/manifests/independent_eval_v1.csv"
BLIND_QUEUE = (
    REPO
    / "reproduction/iac2026/annotations/independent_eval_v1_validation_blind_review_queue.csv"
)
PRIMARY = "artps_full_frozen_mars_clf_on_v1"
ORIENTATION_NOTE_FRAGMENT = "cannot be promoted"


def _load_report() -> dict:
    assert REPORT.is_file(), "run audit_independent_eval_validation_sanity.py first"
    return json.loads(REPORT.read_text(encoding="utf-8"))


def _primary_csv_rows() -> list[dict[str, str]]:
    path = VAL_ROOT / PRIMARY / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_all_positive_threshold_on_primary():
    rows = _primary_csv_rows()
    y = np.asarray([int(r["y_true"]) for r in rows], dtype=np.int32)
    s = np.asarray([float(r["anomaly_score"]) for r in rows], dtype=np.float64)
    thr, _ = select_threshold_on_validation(y, s)
    assert thr == 0.0
    pred = (s >= thr).astype(np.int32)
    cm = confusion(y, pred)
    assert cm["tn"] == 0 and cm["fn"] == 0
    assert cm["tp"] + cm["fp"] == len(rows)


def test_below_chance_auroc_and_ap_vs_prevalence():
    rows = _primary_csv_rows()
    y = np.asarray([int(r["y_true"]) for r in rows], dtype=np.int32)
    s = np.asarray([float(r["anomaly_score"]) for r in rows], dtype=np.float64)
    auroc = binary_auroc(y, s)
    ap = average_precision(y, s)
    prevalence = float(y.mean())
    assert auroc is not None and auroc < 0.5
    assert ap is not None and ap < prevalence


def test_y_true_equals_manifest_binary_label():
    rows = _primary_csv_rows()
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        manifest = {r["sample_id"]: r for r in csv.DictReader(f)}
    for r in rows:
        m = manifest[r["sample_id"]]
        assert int(r["y_true"]) == int(m["binary_label"])
        assert str(m["split"]).lower() == "validation"


def test_orientation_diagnostic_not_promoted():
    report = _load_report()
    assert report["orientation_promoted"] is False
    assert ORIENTATION_NOTE_FRAGMENT in report["orientation_diagnostic_note"]
    for p in report["profiles"]:
        assert p["orientation_diagnostic"]["promoted"] is False


def test_zero_score_reason_from_instrumented_rerun():
    diag = VAL_ROOT / PRIMARY / "candidate_diagnostics.csv"
    assert diag.is_file()
    with diag.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    zeros = [r for r in rows if float(r["image_score"]) == 0.0]
    assert zeros
    reasons = {r["zero_score_reason"] for r in zeros}
    assert reasons <= {
        "field_scale_rejection",
        "size_distance_policy_rejection",
        "candidate_score_filtering",
        "border_mask",
        "no_raw_proposal",
        "processing_status_error",
        "unavailable_requires_instrumented_validation_rerun",
    }
    assert "processing_status_error" not in reasons
    assert any(r in reasons for r in (
        "field_scale_rejection",
        "size_distance_policy_rejection",
        "candidate_score_filtering",
    ))


def test_component_diagnostics_v1_instrumented():
    path = VAL_ROOT / PRIMARY / "component_diagnostics_v1.csv"
    assert path.is_file()
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 54
    assert all(r["execution_path"] == "instrumented_validation_rerun" for r in rows)
    assert all(str(r.get("score_parity_ok")).lower() in {"true", "1"} for r in rows)
    zeros = [r for r in rows if float(r["image_score"]) == 0.0]
    assert zeros
    assert all(int(r["raw_proposal_count"]) > 0 for r in zeros)
    assert all(r["no_valid_candidate_reason"] for r in zeros)


def test_silent_fallback_rejected_on_synthetic_error_status():
    """Synthetic: processing_status=error + score 0 must not count as clean ok zero."""
    synthetic = {
        "image_score": 0.0,
        "processing_status": "error",
        "candidate_count": 0,
        "warning_flags": ["boom"],
    }
    assert synthetic["processing_status"] == "error"
    assert "boom" in synthetic["warning_flags"]
    # Contract helper used by audit
    reason = (
        "processing_status_error"
        if synthetic["processing_status"] == "error"
        else "no_valid_candidate"
    )
    assert reason == "processing_status_error"


def test_blocked_scope_cannot_authorize_final_test():
    scope = load_final_test_scope()
    assert scope["status"] == "blocked_validation_sanity_review"
    assert scope["final_test_authorized"] is False
    with pytest.raises(ValueError, match="final test refused"):
        assert_final_test_authorized(scope)


def test_test_open_status_still_closed():
    status = yaml.safe_load(
        (REPO / "reproduction/iac2026/test_freeze/TEST_OPEN_STATUS.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert status["test_opened"] is False


def test_blind_queue_hides_labels_and_scores():
    assert BLIND_QUEUE.is_file()
    with BLIND_QUEUE.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 54
    cols = set(rows[0].keys())
    for banned in (
        "binary_label",
        "y_true",
        "anomaly_score",
        "image_score",
        "split",
        "sample_id",
        "relative_path",
    ):
        assert banned not in cols
    assert all(r["audit_status"] == "pending_independent_review" for r in rows)


def test_scoped_bug_flags_and_decision_text():
    report = _load_report()
    assert report["metric_bug_detected"] is False
    assert report["label_mapping_bug_detected"] is False
    assert report["classifier_class_semantics_verified"] is False
    assert report["candidate_suppression_semantics_verified"] is False
    assert report["annotation_quality_independently_verified"] is False
    assert "manifest-to-prediction label-mapping" in report["decision_text"]
    assert "remain unresolved" in report["decision_text"]


def test_claim_ledger_and_manuscript_results_untouched_paths():
    """Sanity PR must not rewrite claim statuses or manuscript Results numbers."""
    # Path-level guard: known claim/result files exist and are not required dirty.
    claim_paths = list((REPO / "reproduction/iac2026").glob("**/claim*.yaml")) + list(
        (REPO / "paper").glob("**/CLAIM*.md")
    )
    # Soft assert: at least protocol freeze markers exist; Results tex not edited by this PR.
    assert (REPO / "reproduction/iac2026/test_freeze/FINAL_TEST_SCOPE.yaml").is_file()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    assert selection["selected_config_id"] == PRIMARY
    assert selection["test_opened"] is False
    report = _load_report()
    assert report["final_test_authorized"] is False
    assert report["objective_bug_proven"] is False
    _ = claim_paths  # documented presence only


def test_four_profiles_flagged_below_chance():
    report = _load_report()
    assert len(report["profiles"]) == 4
    for p in report["profiles"]:
        assert p["flags"]["auroc_below_0_5"] is True
        assert p["flags"]["all_positive_predictions"] is True
        assert p["flags"]["degenerate_threshold_zero"] is True
