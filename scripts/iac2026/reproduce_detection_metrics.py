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
    make_run_id,
    read_csv_dicts,
    resolve_repo_path,
    sha256_file,
    write_json,
    write_run_bundle,
    write_text,
)
from _config import ConfigValidationError, load_and_validate_config, require_jsonschema
from audit_reproduction_inputs import AuditResult, audit_inputs
from detection_metrics_lib import (
    average_precision,
    binary_auroc,
    confusion,
    f1_precision_recall,
    map_positive_label,
    orient_scores,
    select_threshold_on_validation,
    trapezoidal_pr_auc,
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


def _verify_audit_freshness(audit: AuditResult, loaded, pred_path: Path, manifest_path: Path) -> List[str]:
    from _common import git_dirty, git_head

    errs: List[str] = []
    if audit.config_sha256 and audit.config_sha256 != loaded.sha256:
        errs.append("stale audit: config_sha256 mismatch")
    if audit.manifest_sha256 and audit.manifest_sha256 != sha256_file(manifest_path):
        errs.append("stale audit: manifest_sha256 mismatch")
    if audit.predictions_sha256 and audit.predictions_sha256 != sha256_file(pred_path):
        errs.append("stale audit: predictions_sha256 mismatch")
    if audit.git_head and audit.git_head != git_head():
        errs.append("stale audit: git_head mismatch")
    if audit.git_dirty is not None and audit.git_dirty != git_dirty():
        errs.append("stale audit: git_dirty mismatch")
    return errs


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reproduce detection metrics from prediction CSV.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--software-verification", action="store_true")
    ap.add_argument("--audit-json", default=None, help="Optional prior audit JSON (re-verified).")
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

    # Audit gate
    if args.audit_json:
        payload = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
        audit = AuditResult(
            passed=bool(payload.get("passed")),
            blockers=list(payload.get("blockers") or []),
            errors=list(payload.get("errors") or []),
            warnings=list(payload.get("warnings") or []),
            config_sha256=payload.get("config_sha256"),
            manifest_sha256=payload.get("manifest_sha256"),
            predictions_sha256=payload.get("predictions_sha256"),
            checkpoint_sha256=payload.get("checkpoint_sha256"),
            git_head=payload.get("git_head"),
            git_dirty=payload.get("git_dirty"),
            evidence_mode=payload.get("evidence_mode"),
        )
        if not audit.passed:
            print("AUDIT JSON reports passed=false", file=sys.stderr)
            return 2
        pred_path = resolve_repo_path(str(cfg["predictions_csv"]))
        man_path = resolve_repo_path(str(cfg["dataset_manifest"]))
        stale = _verify_audit_freshness(audit, loaded, pred_path, man_path)
        if stale:
            print("STALE AUDIT:\n- " + "\n- ".join(stale), file=sys.stderr)
            return 2
    else:
        audit = audit_inputs(loaded, software_verification=args.software_verification)
        if not audit.passed:
            print("AUDIT FAILED — refusing to compute metrics", file=sys.stderr)
            for b in audit.blockers:
                print(f"  blocker: {b}", file=sys.stderr)
            for e in audit.errors:
                print(f"  error: {e}", file=sys.stderr)
            return 2

    run_id = args.run_id or make_run_id("metrics")
    out_dir = resolve_repo_path(str(cfg["output_directory"])) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = resolve_repo_path(str(cfg["predictions_csv"]))
    rows = read_csv_dicts(pred_path)
    y_test_raw, s_test, test_ids = _load_split(rows, str(cfg["test_split"]))
    y_val_raw, s_val, _ = _load_split(rows, str(cfg["validation_split"]))

    y_test = map_positive_label(y_test_raw, cfg["positive_label"])
    y_val = map_positive_label(y_val_raw, cfg["positive_label"]) if y_val_raw.size else y_val_raw
    s_test = orient_scores(s_test, bool(cfg["higher_score_means_more_anomalous"]))
    s_val = orient_scores(s_val, bool(cfg["higher_score_means_more_anomalous"])) if s_val.size else s_val

    incomplete: List[str] = []
    if cfg["task_level"] == "TASK_LEVEL_TBD":
        incomplete.append("task_level_unknown")
    if cfg["pr_metric_method"] == "UNKNOWN":
        incomplete.append("pr_metric_method_unknown")

    policy = str(cfg["threshold_policy"])
    selected_threshold: Optional[float] = None
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
            selected_threshold = float(cfg["fixed_threshold"])
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

    if int(y_test.sum()) + int((y_test == 0).sum()) != int(y_test.size):
        print("internal error: class counts", file=sys.stderr)
        return 2
    if sum(cm.values()) not in (0, int(y_test.size)):
        print("internal error: confusion count", file=sys.stderr)
        return 2

    sw = cfg["evidence_mode"] == "software_verification"
    evidence_class = "software_verification" if sw else "candidate_real_evidence"
    eligible = False  # never auto-close claims from this harness alone

    status = "computed_from_predictions"
    if incomplete:
        status = "reproduction_incomplete:" + ",".join(incomplete)

    metrics: Dict[str, Any] = {
        "claim_ids": cfg["claim_ids"],
        "evidence_class": evidence_class,
        "eligible_for_claim_closure": eligible,
        "task_level": cfg["task_level"],
        "score_semantics": cfg["score_semantics"],
        "higher_score_means_more_anomalous": bool(cfg["higher_score_means_more_anomalous"]),
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
            "Accepted-abstract targets are not pass/fail criteria."
        ),
        "test_sample_ids": test_ids,
        "model_name": cfg["model_name"],
        "model_version": cfg["model_version"],
        "config_id": str(cfg.get("config_id", cfg["model_name"])),
    }

    require_jsonschema()
    from _common import load_json_schema
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
            "accepted_abstract_targets_not_used_as_pass_fail": True,
            "evidence_class": evidence_class,
            "eligible_for_claim_closure": eligible,
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
                f"- eligible_for_claim_closure: `{eligible}`",
                f"- status: `{status}`",
                f"- AUROC: {auroc}",
                f"- average_precision: {ap}",
                f"- trapezoidal_pr_auc: {trap}",
                f"- primary: {primary_name}={primary_val}",
                f"- F1: {f1}",
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
