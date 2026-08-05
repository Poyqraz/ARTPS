"""Compute detection metrics from a pinned prediction table (audit-gated)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    load_json_schema,
    make_run_id,
    read_csv_dicts,
    resolve_repo_path,
    sha256_file,
    write_json,
    write_run_bundle,
    write_text,
)
from _config import (
    PIXEL_REGION_MSG,
    ConfigValidationError,
    load_and_validate_config,
    require_jsonschema,
    validate_instance,
)
from audit_reproduction_inputs import AuditResult, audit_inputs
from detection_metrics_lib import (
    average_precision,
    binary_auroc,
    canonical_threshold,
    confusion,
    f1_precision_recall,
    map_positive_label,
    orient_scores,
    score_orientation_meta,
    select_threshold_on_validation,
    trapezoidal_pr_auc,
)


COMPARE_FIELDS = (
    "passed",
    "evidence_mode",
    "claim_ids",
    "config_id",
    "config_sha256",
    "manifest_sha256",
    "predictions_sha256",
    "checkpoint_sha256",
    "git_head",
    "git_dirty",
    "protocol_id",
    "protocol_lock_sha256",
    "evaluation_purpose",
    "annotation_version",
)


def _load_split(
    rows: Sequence[Dict[str, str]], split: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ids: List[str] = []
    y: List[int] = []
    s: List[float] = []
    for r in rows:
        if r.get("split") != split:
            continue
        ids.append(r["sample_id"])
        y.append(int(r["y_true"]))
        s.append(float(r["anomaly_score"]))
    return np.asarray(y, dtype=np.int32), np.asarray(s, dtype=np.float64), ids


def _compare_prior_audit(prior: Dict[str, Any], fresh: AuditResult, *, real: bool) -> List[str]:
    errs: List[str] = []
    schema_errs = validate_instance(prior, "input_audit.schema.json")
    if schema_errs:
        return ["prior audit schema validation failed: " + "; ".join(schema_errs)]

    fresh_keys = fresh.compare_keys()
    for key in COMPARE_FIELDS:
        if key not in prior:
            errs.append(f"prior audit missing field {key}")
            continue
        if prior[key] != fresh_keys[key]:
            errs.append(f"prior audit {key} mismatch: prior={prior[key]!r} fresh={fresh_keys[key]!r}")

    if real:
        for key in (
            "config_sha256",
            "manifest_sha256",
            "predictions_sha256",
            "checkpoint_sha256",
            "git_head",
            "protocol_id",
            "protocol_lock_sha256",
            "evaluation_purpose",
            "annotation_version",
        ):
            val = prior.get(key)
            if key in ("protocol_id", "protocol_lock_sha256", "annotation_version"):
                # Required non-null for independent real runs; historical may be null
                if prior.get("evaluation_purpose") == "current_reproducible_evaluation":
                    if val is None or val == "":
                        errs.append(f"real_evidence prior audit {key} must be non-null")
                continue
            if val is None or val == "" or (isinstance(val, str) and len(val) < 7):
                errs.append(f"real_evidence prior audit {key} must be non-null")
        if prior.get("git_dirty") is None:
            errs.append("real_evidence prior audit git_dirty must be non-null")
    return errs


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reproduce detection metrics from prediction CSV.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--software-verification", action="store_true")
    ap.add_argument(
        "--audit-json",
        default=None,
        help="Optional prior audit JSON for comparison only (never authoritative).",
    )
    args = ap.parse_args(argv)

    try:
        loaded = load_and_validate_config(args.config)
    except ConfigValidationError as exc:
        print(f"CONFIG VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2

    cfg = loaded.data
    if cfg["evidence_mode"] == "real_evidence" and args.software_verification:
        print("real_evidence forbids --software-verification", file=sys.stderr)
        return 2
    if cfg["evidence_mode"] == "software_verification" and not args.software_verification:
        print(
            "software_verification config requires --software-verification",
            file=sys.stderr,
        )
        return 2

    task = cfg["task_level"]
    if task in ("pixel_binary", "region_binary"):
        print(PIXEL_REGION_MSG, file=sys.stderr)
        return 2

    # Always run a fresh audit — prior JSON cannot bypass this.
    fresh = audit_inputs(loaded, software_verification=args.software_verification)
    if not fresh.passed:
        print("AUDIT FAILED — refusing to compute metrics", file=sys.stderr)
        for b in fresh.blockers:
            print(f"  blocker: {b}", file=sys.stderr)
        for e in fresh.errors:
            print(f"  error: {e}", file=sys.stderr)
        return 2

    if args.audit_json:
        prior = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
        cmp_errs = _compare_prior_audit(
            prior, fresh, real=cfg["evidence_mode"] == "real_evidence"
        )
        if cmp_errs:
            print("PRIOR AUDIT COMPARISON FAILED:\n- " + "\n- ".join(cmp_errs), file=sys.stderr)
            return 2

    audit = fresh

    run_id = args.run_id or make_run_id("metrics")
    out_dir = resolve_repo_path(str(cfg["output_directory"])) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = resolve_repo_path(str(cfg["predictions_csv"]))
    rows = read_csv_dicts(pred_path)
    y_test_raw, s_test_raw, test_ids = _load_split(rows, str(cfg["test_split"]))
    y_val_raw, s_val_raw, _ = _load_split(rows, str(cfg["validation_split"]))

    higher = bool(cfg["higher_score_means_more_anomalous"])
    orient_meta = score_orientation_meta(higher)
    y_test = map_positive_label(y_test_raw, cfg["positive_label"])
    y_val = map_positive_label(y_val_raw, cfg["positive_label"]) if y_val_raw.size else y_val_raw
    s_test = orient_scores(s_test_raw, higher)
    s_val = orient_scores(s_val_raw, higher) if s_val_raw.size else s_val_raw

    incomplete: List[str] = []
    if cfg["task_level"] == "TASK_LEVEL_TBD":
        incomplete.append("task_level_unknown")
    if cfg["pr_metric_method"] == "UNKNOWN":
        incomplete.append("pr_metric_method_unknown")

    policy = str(cfg["threshold_policy"])
    selected_threshold: Optional[float] = None
    raw_fixed: Optional[float] = None
    threshold_source = policy
    val_metric_at: Optional[float] = None
    sel_split = None
    sel_metric = None
    sel_tie = None

    if policy == "fixed_historical":
        if cfg.get("fixed_threshold") is None:
            incomplete.append("threshold_policy_unknown")
            threshold_source = "fixed_historical_missing_value"
        else:
            raw_fixed = float(cfg["fixed_threshold"])
            selected_threshold = canonical_threshold(raw_fixed, higher)
            threshold_source = "fixed_historical"
    elif policy == "validation_selected":
        sel_split = str(cfg["validation_split"])
        sel_metric = str(cfg["threshold_selection_metric"])
        sel_tie = str(cfg["threshold_tie_break"])
        selected_threshold, val_metric_at = select_threshold_on_validation(
            y_val, s_val, metric=sel_metric, tie_break=sel_tie
        )
        if selected_threshold is None:
            incomplete.append("threshold_policy_unknown")
            threshold_source = "validation_selected_failed"
        else:
            threshold_source = "validation_selected"
    else:
        incomplete.append("threshold_policy_unknown")
        selected_threshold = None

    auroc = binary_auroc(y_test, s_test) if y_test.size else None
    ap = average_precision(y_test, s_test) if y_test.size else None
    trap = trapezoidal_pr_auc(y_test, s_test) if y_test.size else None

    primary_name = str(cfg["pr_metric_method"])
    if primary_name == "average_precision":
        primary_val = ap
    elif primary_name == "trapezoidal_pr_auc":
        primary_val = trap
    else:
        primary_val = None

    f1 = precision = recall = None
    cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    if selected_threshold is not None and y_test.size:
        y_pred = (s_test >= selected_threshold).astype(np.int32)
        cm = confusion(y_test, y_pred)
        f1, precision, recall = f1_precision_recall(cm)

    if sum(cm.values()) not in (0, int(y_test.size)):
        print("internal error: confusion count", file=sys.stderr)
        return 2

    sw = cfg["evidence_mode"] == "software_verification"
    evidence_class = "software_verification" if sw else "candidate_real_evidence"
    status = "computed_from_predictions"
    if incomplete:
        status = "reproduction_incomplete:" + ",".join(incomplete)

    independent = cfg.get("evaluation_purpose") == "current_reproducible_evaluation"
    metrics: Dict[str, Any] = {
        "claim_ids": cfg["claim_ids"],
        "evidence_class": evidence_class,
        "eligible_for_claim_closure": False,
        "eligible_for_C05_C06_closure": False,
        "historical_claim_reproduction": False if independent else (
            cfg.get("evaluation_purpose") == "historical_claim_reproduction"
        ),
        "eligible_for_IND_EVAL_V1_result_reporting": False,
        "author_verified": False,
        "protocol_id": cfg.get("protocol_id"),
        "protocol_lock_sha256": cfg.get("protocol_lock_sha256"),
        "evaluation_purpose": cfg.get("evaluation_purpose"),
        "annotation_version": cfg.get("annotation_version"),
        "image_score_aggregation": cfg.get("image_score_aggregation"),
        "task_level": cfg["task_level"],
        "score_semantics": cfg["score_semantics"],
        "higher_score_means_more_anomalous": higher,
        "raw_score_orientation": orient_meta["raw_score_orientation"],
        "canonical_score_transform": orient_meta["canonical_score_transform"],
        "raw_fixed_threshold": raw_fixed,
        "canonical_selected_threshold": selected_threshold,
        "decision_operator": orient_meta["decision_operator"],
        "test_sample_count": int(y_test.size),
        "positive_count": int(y_test.sum()) if y_test.size else 0,
        "negative_count": int((y_test == 0).sum()) if y_test.size else 0,
        "auroc": auroc,
        "average_precision": ap,
        "trapezoidal_pr_auc": trap,
        "primary_pr_metric_name": primary_name,
        "primary_pr_metric_value": primary_val,
        "pr_metric_method": primary_name,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "selected_threshold": selected_threshold,
        "threshold_source": threshold_source,
        "threshold_selection_split": sel_split,
        "threshold_selection_metric": sel_metric,
        "threshold_tie_break": sel_tie,
        "validation_metric_at_selected_threshold": val_metric_at,
        "confusion_matrix": cm,
        "bootstrap_ci": None,
        "reproduction_status": status,
        "notes": (
            "Neutral report from prediction table. "
            "Accepted-abstract targets are not pass/fail criteria. "
            "Independent eval outputs are not C05/C06 closures."
        ),
        "test_sample_ids": test_ids,
        "model_name": cfg["model_name"],
        "model_version": cfg["model_version"],
        "config_id": str(cfg["config_id"]),
        "manifest_sha256": audit.manifest_sha256,
        "predictions_sha256": sha256_file(pred_path),
        "checkpoint_sha256": audit.checkpoint_sha256,
        "git_head": audit.git_head,
        "git_dirty": audit.git_dirty,
    }

    require_jsonschema()
    import jsonschema

    jsonschema.Draft202012Validator(load_json_schema("detection_metrics.schema.json")).validate(metrics)

    shutil.copy2(pred_path, out_dir / "predictions.csv")
    write_json(out_dir / "detection_metrics.json", metrics)
    write_json(out_dir / "input_audit.json", audit.to_dict())
    write_run_bundle(
        out_dir,
        config_path=loaded.path,
        command=sys.argv,
        provenance_extra={
            "predictions_sha256": sha256_file(pred_path),
            "manifest_sha256": audit.manifest_sha256,
            "checkpoint_sha256": audit.checkpoint_sha256,
            "protocol_id": metrics["protocol_id"],
            "protocol_lock_sha256": metrics["protocol_lock_sha256"],
            "evaluation_purpose": metrics["evaluation_purpose"],
            "annotation_version": metrics["annotation_version"],
            "image_score_aggregation": metrics["image_score_aggregation"],
            "historical_claim_reproduction": metrics["historical_claim_reproduction"],
            "eligible_for_C05_C06_closure": False,
            "eligible_for_IND_EVAL_V1_result_reporting": False,
            "accepted_abstract_targets_not_used_as_pass_fail": True,
            "evidence_class": evidence_class,
            "eligible_for_claim_closure": False,
            "author_verified": False,
            "sklearn_version": _sklearn_version(),
        },
    )
    write_text(
        out_dir / "detection_metrics.md",
        "\n".join(
            [
                f"# Detection metrics `{run_id}`",
                "",
                f"- evidence_class: `{evidence_class}`",
                f"- evaluation_purpose: `{metrics.get('evaluation_purpose')}`",
                f"- protocol_id: `{metrics.get('protocol_id')}`",
                f"- historical_claim_reproduction: `{metrics.get('historical_claim_reproduction')}`",
                f"- eligible_for_C05_C06_closure: `False`",
                f"- eligible_for_claim_closure: `False`",
                f"- eligible_for_IND_EVAL_V1_result_reporting: `False`",
                f"- status: `{status}`",
                f"- AUROC: {auroc}",
                f"- average_precision: {ap}",
                f"- F1: {f1}",
                f"- decision_operator: {orient_meta['decision_operator']}",
                "",
                "Accepted-abstract numbers are not used as pass/fail criteria.",
                "",
            ]
        ),
    )
    print(f"Wrote {out_dir / 'detection_metrics.json'}")
    print(f"status={status} auroc={auroc} ap={ap} f1={f1} evidence_class={evidence_class}")
    return 0


def _sklearn_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("scikit-learn")
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
