"""Audit independent_eval_v1 validation predictions for sanity (CSV/JSONL only; no GPU)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "iac2026"))

from detection_metrics_lib import (  # noqa: E402
    average_precision,
    binary_auroc,
    confusion,
    f1_precision_recall,
    select_threshold_on_validation,
)

VAL_ROOT = REPO_ROOT / "results" / "iac2026" / "independent_eval_v1" / "validation"
MANIFEST = REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1.csv"
SELECTION = VAL_ROOT / "profile_selection.json"
REPORT_JSON = VAL_ROOT / "validation_sanity_report.json"
REPORT_MD = (
    REPO_ROOT / "paper" / "iac2026" / "reproduction" / "INDEPENDENT_EVAL_V1_VALIDATION_SANITY.md"
)
ORIENTATION_NOTE = (
    "Negated-score metrics are diagnostic and cannot be promoted unless the production "
    "score contract is objectively demonstrated to have the opposite orientation."
)
METRIC_TOL = 1e-6
BLIND_QUEUE_SEED = 20260806


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_included_resolved(row: dict[str, str]) -> bool:
    incl = str(row.get("inclusion_status") or row.get("included") or "").strip().lower()
    adj = str(row.get("adjudication_status") or row.get("resolved") or "").strip().lower()
    return incl in {"included", "1", "true", "yes"} and adj in {"resolved", "1", "true", "yes"}


def _sklearn_auroc_ap(
    y: np.ndarray, scores: np.ndarray
) -> tuple[float | None, float | None, str | None]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception as exc:  # ImportError or binary/numpy ABI failures
        return None, None, f"sklearn_unavailable:{type(exc).__name__}"
    if int(y.sum()) == 0 or int(y.sum()) == len(y):
        return None, None, "degenerate_labels"
    try:
        return (
            float(roc_auc_score(y, scores)),
            float(average_precision_score(y, scores)),
            None,
        )
    except Exception as exc:
        return None, None, f"sklearn_compute_failed:{type(exc).__name__}"


def _finite_close(a: float | None, b: float | None, tol: float = METRIC_TOL) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def analyze_profile(
    config_id: str,
    predictions_csv: Path,
    selection_metrics: dict[str, Any] | None,
    manifest_by_id: dict[str, dict[str, str]],
    val_ids: set[str],
) -> dict[str, Any]:
    rows = _read_csv(predictions_csv)
    jsonl_path = predictions_csv.with_name("predictions.jsonl")
    jsonl_rows = _read_jsonl(jsonl_path) if jsonl_path.is_file() else []
    jsonl_by_id = {str(r.get("sample_id")): r for r in jsonl_rows}

    sample_ids = [r["sample_id"] for r in rows]
    dup_ids = [sid for sid, c in Counter(sample_ids).items() if c > 1]
    y = np.asarray([int(r["y_true"]) for r in rows], dtype=np.int32)
    scores = np.asarray([float(r["anomaly_score"]) for r in rows], dtype=np.float64)

    n = int(len(rows))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    prevalence = float(n_pos / n) if n else float("nan")

    custom_auroc = binary_auroc(y, scores)
    custom_ap = average_precision(y, scores)
    sk_auroc, sk_ap, sk_err = _sklearn_auroc_ap(y, scores)

    thr, thr_f1 = select_threshold_on_validation(y, scores)
    pred = (scores >= float(thr if thr is not None else 0.0)).astype(np.int32)
    cm = confusion(y, pred)
    f1, prec, rec = f1_precision_recall(cm)

    mean_pos = float(scores[y == 1].mean()) if n_pos else float("nan")
    mean_neg = float(scores[y == 0].mean()) if n_neg else float("nan")
    median_pos = float(np.median(scores[y == 1])) if n_pos else float("nan")
    median_neg = float(np.median(scores[y == 0])) if n_neg else float("nan")

    zero_mask = scores == 0.0
    n_zero = int(zero_mask.sum())
    unique_scores = int(np.unique(scores).size)
    score_min = float(scores.min()) if n else float("nan")
    score_max = float(scores.max()) if n else float("nan")
    score_std = float(scores.std()) if n else float("nan")

    neg_scores = -scores
    auroc_neg = binary_auroc(y, neg_scores)
    ap_neg = average_precision(y, neg_scores)
    one_minus = None
    auroc_om = None
    ap_om = None
    if score_min >= 0.0 and score_max <= 1.0:
        one_minus = 1.0 - scores
        auroc_om = binary_auroc(y, one_minus)
        ap_om = average_precision(y, one_minus)

    metric_bug = False
    metric_notes: list[str] = []
    if sk_err:
        metric_notes.append(sk_err)
    else:
        if not _finite_close(custom_auroc, sk_auroc):
            metric_bug = True
            metric_notes.append(f"auroc_mismatch custom={custom_auroc} sklearn={sk_auroc}")
        if not _finite_close(custom_ap, sk_ap):
            metric_bug = True
            metric_notes.append(f"ap_mismatch custom={custom_ap} sklearn={sk_ap}")

    y_true_mismatches: list[str] = []
    missing_manifest: list[str] = []
    for r in rows:
        sid = r["sample_id"]
        m = manifest_by_id.get(sid)
        if m is None:
            missing_manifest.append(sid)
            continue
        if int(r["y_true"]) != int(m["binary_label"]):
            y_true_mismatches.append(sid)

    pred_ids = set(sample_ids)
    id_set_match = pred_ids == val_ids

    zero_reasons: Counter[str] = Counter()
    error_count = 0
    for r in rows:
        if float(r["anomaly_score"]) != 0.0:
            continue
        jl = jsonl_by_id.get(r["sample_id"], {})
        status = str(jl.get("processing_status") or "missing_jsonl")
        if status == "error":
            zero_reasons["processing_status_error"] += 1
            error_count += 1
        elif int(jl.get("candidate_count") or 0) == 0 and status == "ok":
            zero_reasons["no_valid_candidate"] += 1
        elif jl.get("warning_flags"):
            zero_reasons["ok_with_warnings"] += 1
        else:
            zero_reasons["other_or_missing_jsonl"] += 1

    flags = {
        "all_positive_predictions": cm["tn"] == 0
        and cm["fn"] == 0
        and cm["fp"] + cm["tp"] == n,
        "auroc_below_0_5": custom_auroc is not None and custom_auroc < 0.5,
        "ap_below_positive_prevalence": custom_ap is not None and custom_ap < prevalence,
        "score_collapse": unique_scores <= 2 or score_std < 1e-6,
        "excessive_zero_scores": n_zero >= max(1, n // 4),
        "class_score_order_reversed": n_pos > 0 and n_neg > 0 and mean_pos < mean_neg,
        "degenerate_threshold_zero": thr is not None and float(thr) == 0.0,
        "metric_implementation_bug": metric_bug,
        "y_true_mapping_bug": bool(y_true_mismatches or missing_manifest or not id_set_match),
        "duplicate_sample_ids": bool(dup_ids),
        "processing_errors_present": error_count > 0,
        "n_not_54": n != 54,
        "not_balanced_27_27": not (n_pos == 27 and n_neg == 27),
    }

    return {
        "config_id": config_id,
        "predictions_csv": str(predictions_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "positive_prevalence": prevalence,
        "positive_label": 1,
        "higher_score_means_more_anomalous": True,
        "decision_rule": "score >= threshold",
        "validation_id_set_match": id_set_match,
        "score_stats": {
            "min": score_min,
            "max": score_max,
            "mean": float(scores.mean()) if n else float("nan"),
            "std": score_std,
            "unique_count": unique_scores,
            "zero_count": n_zero,
            "zero_fraction": float(n_zero / n) if n else float("nan"),
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "median_pos": median_pos,
            "median_neg": median_neg,
        },
        "metrics_custom": {
            "auroc": custom_auroc,
            "average_precision": custom_ap,
            "selected_threshold": thr,
            "f1_at_selected_threshold": thr_f1,
            "confusion_matrix": cm,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        },
        "metrics_sklearn": {
            "auroc": sk_auroc,
            "average_precision": sk_ap,
            "error": sk_err,
        },
        "orientation_diagnostic": {
            "note": ORIENTATION_NOTE,
            "promoted": False,
            "raw": {"auroc": custom_auroc, "average_precision": custom_ap},
            "negated": {"auroc": auroc_neg, "average_precision": ap_neg},
            "one_minus": (
                {"auroc": auroc_om, "average_precision": ap_om}
                if one_minus is not None
                else None
            ),
        },
        "selection_artifact_metrics": selection_metrics,
        "zero_score_reasons": dict(zero_reasons),
        "y_true_audit": {
            "mismatches": y_true_mismatches,
            "missing_in_manifest": missing_manifest,
        },
        "duplicate_sample_ids": dup_ids,
        "flags": flags,
        "metric_crosscheck_notes": metric_notes,
    }


def write_candidate_diagnostics(config_id: str) -> Path:
    """Extract candidate/zero diagnostics from committed JSONL (no re-inference)."""
    out_dir = VAL_ROOT / config_id
    jsonl_path = out_dir / "predictions.jsonl"
    out_csv = out_dir / "candidate_diagnostics.csv"
    rows = _read_jsonl(jsonl_path)
    unavailable = "unavailable_in_committed_jsonl"
    fieldnames = [
        "sample_id",
        "split",
        "image_score",
        "candidate_count",
        "valid_candidate_count",
        "top_candidate_score",
        "anomaly_mse",
        "processing_status",
        "warning_flags",
        "zero_score_reason",
        "classifier_score",
        "mask_fraction",
        "notes",
    ]
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        score = float(r.get("image_score") or 0.0)
        status = str(r.get("processing_status") or "")
        cand = int(r.get("candidate_count") or 0)
        if score == 0.0:
            if status == "error":
                reason = "processing_status_error"
            elif cand == 0 and status == "ok":
                reason = "no_valid_candidate"
            elif r.get("warning_flags"):
                reason = "ok_with_warnings"
            else:
                reason = "other"
        else:
            reason = ""
        out_rows.append(
            {
                "sample_id": r.get("sample_id"),
                "split": r.get("split"),
                "image_score": score,
                "candidate_count": cand,
                "valid_candidate_count": r.get("valid_candidate_count"),
                "top_candidate_score": r.get("top_candidate_score"),
                "anomaly_mse": r.get("anomaly_mse"),
                "processing_status": status,
                "warning_flags": "|".join(r.get("warning_flags") or []),
                "zero_score_reason": reason,
                "classifier_score": "",
                "mask_fraction": "",
                "notes": unavailable,
            }
        )
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    return out_csv


def write_blind_review_queue(
    manifest_rows: list[dict[str, str]], seed: int = BLIND_QUEUE_SEED
) -> Path:
    out = (
        REPO_ROOT
        / "reproduction"
        / "iac2026"
        / "annotations"
        / "independent_eval_v1_validation_blind_review_queue.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    val = [
        r
        for r in manifest_rows
        if str(r.get("split")).strip().lower() == "validation" and _is_included_resolved(r)
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(val))
    fieldnames = [
        "review_order",
        "sample_id",
        "relative_path",
        "path_exists",
        "audit_status",
        "seed",
    ]
    rows_out: list[dict[str, Any]] = []
    for i, idx in enumerate(order):
        r = val[int(idx)]
        rel = r.get("relative_path") or ""
        path = REPO_ROOT / rel if rel and not Path(rel).is_absolute() else Path(rel)
        rows_out.append(
            {
                "review_order": i,
                "sample_id": r["sample_id"],
                "relative_path": rel,
                "path_exists": bool(path.is_file()) if rel else False,
                "audit_status": "pending_independent_review",
                "seed": seed,
            }
        )
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    return out


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Independent Eval V1 — Validation Sanity Audit",
        "",
        "Status: **blocked_validation_sanity_review**. Final test remains closed.",
        "",
        ORIENTATION_NOTE,
        "",
        f"- Protocol: `{report['protocol_id']}`",
        f"- Selected config (historical, immutable): `{report['selected_config_id']}`",
        f"- Profiles audited: {len(report['profiles'])}",
        f"- Objective bug proven: `{report['objective_bug_proven']}`",
        f"- Final test authorized: `{report['final_test_authorized']}`",
        "",
        "## Profile summary",
        "",
        "| config_id | n | AUROC | AP | thr | CM (tn/fp/fn/tp) | flags |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for p in report["profiles"]:
        m = p["metrics_custom"]
        cm = m["confusion_matrix"]
        active_flags = [k for k, v in p["flags"].items() if v]
        auroc = m["auroc"]
        ap = m["average_precision"]
        lines.append(
            f"| `{p['config_id']}` | {p['n']} | "
            f"{auroc:.4f} | {ap:.4f} | {m['selected_threshold']} | "
            f"{cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']} | "
            f"{', '.join(active_flags) or 'none'} |"
        )
    lines.extend(["", "## Blockers", ""])
    for b in report["blockers"]:
        lines.append(f"- `{b}`")
    lines.extend(["", "## Decision", "", report["decision_text"], ""])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-candidate-diagnostics", action="store_true")
    parser.add_argument("--skip-blind-queue", action="store_true")
    args = parser.parse_args(argv)

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    manifest_rows = _read_csv(MANIFEST)
    manifest_by_id = {r["sample_id"]: r for r in manifest_rows}

    val_ids = {
        r["sample_id"]
        for r in manifest_rows
        if str(r.get("split")).strip().lower() == "validation" and _is_included_resolved(r)
    }

    profiles_out: list[dict[str, Any]] = []
    for cand in selection.get("candidates", []):
        cid = cand["config_id"]
        pred_path = REPO_ROOT / cand["predictions_csv"]
        profile = analyze_profile(
            cid,
            pred_path,
            cand.get("metrics"),
            manifest_by_id,
            val_ids,
        )
        profiles_out.append(profile)
        if (
            not args.skip_candidate_diagnostics
            and cid == selection.get("selected_config_id")
        ):
            write_candidate_diagnostics(cid)

    if not args.skip_blind_queue:
        write_blind_review_queue(manifest_rows)

    any_metric_bug = any(p["flags"]["metric_implementation_bug"] for p in profiles_out)
    any_label_bug = any(p["flags"]["y_true_mapping_bug"] for p in profiles_out)
    objective_bug = bool(any_metric_bug or any_label_bug)

    blockers = [
        "validation_auroc_below_chance",
        "validation_ap_below_balanced_prevalence",
        "degenerate_all_positive_threshold",
        "score_orientation_not_verified",
        "label_score_semantics_not_verified",
    ]
    if any_metric_bug:
        blockers.append("metric_implementation_bug")
    if any_label_bug:
        blockers.append("y_true_mapping_bug")

    decision_text = (
        "No objective implementation bug proven from CSV/JSONL cross-checks; "
        "below-chance ranking and all-positive threshold=0.0 remain validation "
        "sanity blockers. `profile_selection.json` stays immutable. "
        "`final_test_authorized=false`; do not open the test split."
    )
    if objective_bug:
        decision_text = (
            "Objective bug flag(s) set — see profile flags. Corrected config "
            "`…_v1_1` and revalidation are required before any final-test authorization; "
            "this audit does not promote orientation transforms."
        )

    report = {
        "protocol_id": "independent_eval_v1",
        "status": "blocked_validation_sanity_review",
        "selected_config_id": selection.get("selected_config_id"),
        "selection_artifact": str(SELECTION.relative_to(REPO_ROOT)).replace("\\", "/"),
        "orientation_diagnostic_note": ORIENTATION_NOTE,
        "orientation_promoted": False,
        "objective_bug_proven": objective_bug,
        "final_test_authorized": False,
        "blockers": blockers,
        "profiles": profiles_out,
        "decision_text": decision_text,
        "asserts": {
            "positive_label": 1,
            "higher_score_means_more_anomalous": True,
            "decision_operator": "score >= t",
            "expected_n": 54,
            "expected_balance": "27/27",
        },
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
