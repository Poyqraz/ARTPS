"""Compute AUROC / AUPRC / F1 from a pinned prediction table (no test-set threshold search)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Package-free import when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    REPO_ROOT,
    copy_config_sidecar,
    environment_snapshot,
    git_dirty,
    load_yaml,
    make_run_id,
    read_csv_dicts,
    resolve_repo_path,
    sha256_file,
    write_json,
    write_text,
)


def _binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y = y_true.astype(np.int32)
    s = scores.astype(np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    # Mann–Whitney / rank form
    order = np.argsort(s)
    ranks = np.empty_like(s, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # Average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos_ranks = float(ranks[y == 1].sum())
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    return (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _binary_auprc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y = y_true.astype(np.int32)
    s = scores.astype(np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / float(y.sum())
    precision = tp / np.maximum(tp + fp, 1)
    # Append (0,1) start for PR curve integration
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(precision, recall))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _f1_precision_recall(cm: Dict[str, int]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return f1, precision, recall


def _select_threshold_on_validation(
    y_val: np.ndarray, scores_val: np.ndarray
) -> float:
    """Choose threshold maximizing F1 on validation only (never on test)."""
    candidates = np.unique(scores_val.astype(np.float64))
    if candidates.size == 0:
        return 0.5
    best_t = float(candidates[0])
    best_f1 = -1.0
    for t in candidates:
        pred = (scores_val >= t).astype(np.int32)
        f1, _, _ = _f1_precision_recall(_confusion(y_val, pred))
        score = -1.0 if f1 is None else float(f1)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


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


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reproduce detection metrics from prediction CSV.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--skip-audit-gate",
        action="store_true",
        help="Dev only: skip requiring a prior passing audit (CI dry-runs may use fixtures carefully).",
    )
    args = ap.parse_args(argv)

    config_path = resolve_repo_path(args.config)
    cfg = load_yaml(config_path)
    run_id = args.run_id or make_run_id("metrics")
    out_dir = resolve_repo_path(cfg.get("output_directory", "results/iac2026/reproduction")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    task_level = str(cfg.get("task_level", "TASK_LEVEL_TBD"))
    threshold_policy = str(cfg.get("threshold_policy", "unknown")).lower()
    reproduction_incomplete: List[str] = []

    if task_level.upper().endswith("_TBD") or task_level == "TASK_LEVEL_TBD":
        reproduction_incomplete.append("task_level_unknown")

    pred_path = resolve_repo_path(str(cfg["predictions_csv"]))
    rows = read_csv_dicts(pred_path)

    y_test, s_test, test_ids = _load_split(rows, str(cfg.get("test_split", "test")))
    y_val, s_val, _ = _load_split(rows, str(cfg.get("validation_split", "validation")))

    selected_threshold: Optional[float] = None
    threshold_source = threshold_policy

    if threshold_policy == "fixed_historical":
        if cfg.get("fixed_threshold") is None:
            reproduction_incomplete.append("threshold_policy_unknown")
            threshold_source = "fixed_historical_missing_value"
        else:
            selected_threshold = float(cfg["fixed_threshold"])
            threshold_source = "fixed_historical"
    elif threshold_policy == "validation_selected":
        if y_val.size == 0:
            reproduction_incomplete.append("threshold_policy_unknown")
            threshold_source = "validation_selected_but_no_val_rows"
        else:
            selected_threshold = _select_threshold_on_validation(y_val, s_val)
            threshold_source = "validation_selected"
    else:
        reproduction_incomplete.append("threshold_policy_unknown")
        threshold_source = "unknown"
        # Do not search on test. Leave F1 null.
        selected_threshold = None

    auroc = _binary_auroc(y_test, s_test) if y_test.size else None
    auprc = _binary_auprc(y_test, s_test) if y_test.size else None

    f1 = precision = recall = None
    cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    if selected_threshold is not None and y_test.size:
        y_pred = (s_test >= selected_threshold).astype(np.int32)
        cm = _confusion(y_test, y_pred)
        f1, precision, recall = _f1_precision_recall(cm)

    status = "computed_from_predictions"
    if reproduction_incomplete:
        status = "reproduction_incomplete:" + ",".join(reproduction_incomplete)

    metrics: Dict[str, Any] = {
        "claim_ids": cfg.get("claim_ids", ["C05"]),
        "test_sample_count": int(y_test.size),
        "positive_count": int(y_test.sum()) if y_test.size else 0,
        "negative_count": int((y_test == 0).sum()) if y_test.size else 0,
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "selected_threshold": selected_threshold,
        "threshold_source": threshold_source,
        "confusion_matrix": cm,
        "bootstrap_ci": None,
        "reproduction_status": status,
        "notes": (
            "Neutral report of metrics from the prediction table. "
            "Does not compare to accepted-abstract targets. "
            f"task_level={task_level}"
        ),
        "test_sample_ids": test_ids,
    }

    # Run bundle
    copy_config_sidecar(config_path, out_dir)
    shutil.copy2(pred_path, out_dir / "predictions.csv")
    write_json(out_dir / "detection_metrics.json", metrics)
    write_json(out_dir / "environment.json", environment_snapshot())
    write_json(
        out_dir / "provenance.json",
        {
            "git_head": environment_snapshot()["git_head"],
            "git_dirty": git_dirty(),
            "predictions_sha256": sha256_file(pred_path),
            "config_sha256": sha256_file(config_path),
            "command": " ".join(sys.argv),
            "accepted_abstract_targets_not_used_as_pass_fail": True,
        },
    )
    write_text(
        out_dir / "command.txt",
        " ".join(sys.argv) + "\n",
    )

    # Human-readable summary — never pass/fail against 0.894
    lines = [
        f"# Detection metrics `{run_id}`",
        "",
        f"- status: `{status}`",
        f"- test_n: {metrics['test_sample_count']}",
        f"- AUROC: {auroc}",
        f"- AUPRC: {auprc}",
        f"- F1: {f1}",
        f"- threshold: {selected_threshold} ({threshold_source})",
        "",
        "Accepted-abstract numbers are not used as pass/fail criteria.",
    ]
    write_text(out_dir / "detection_metrics.md", "\n".join(lines) + "\n")

    print(f"Wrote {out_dir / 'detection_metrics.json'}")
    print(f"status={status} auroc={auroc} auprc={auprc} f1={f1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
