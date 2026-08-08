"""Remap frozen ARTPS validation scores onto independent_eval_v1_1 labels.

Uses the already-committed FP32 predictions for artps_full_frozen_mars_clf_on_v1.
Does NOT re-run inference, does NOT load test images, does NOT mutate
profile_selection.json or independent_eval_v1. Fail-closed if test rows appear
or if validation is single-class.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "iac2026"))

from build_independent_eval_split import check_group_leakage  # noqa: E402
from detection_metrics_lib import (  # noqa: E402
    average_precision,
    binary_auroc,
    confusion,
    f1_precision_recall,
    orient_scores,
    select_threshold_on_validation,
)

V1_MANIFEST = REPO_ROOT / "reproduction/iac2026/manifests/independent_eval_v1.csv"
V11_MANIFEST = REPO_ROOT / "reproduction/iac2026/manifests/independent_eval_v1_1.csv"
FROZEN_PRED = (
    REPO_ROOT
    / "results/iac2026/independent_eval_v1/validation"
    / "artps_full_frozen_mars_clf_on_v1/predictions.csv"
)
PROFILE_SELECTION = (
    REPO_ROOT / "results/iac2026/independent_eval_v1/validation/profile_selection.json"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "results/iac2026/independent_eval_v1_1/validation"
    / "artps_full_frozen_mars_clf_on_v1_relabel"
)
FROZEN_CONFIG_ID = "artps_full_frozen_mars_clf_on_v1"
GROUP_FIELDS = ["sha256", "duplicate_group_id", "scene_group_id", "source_id", "sequence_id"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_integrity(v11: list[dict[str, str]], v1: list[dict[str, str]]) -> dict[str, Any]:
    v1_by = {r["sample_id"]: r for r in v1}
    if set(v1_by) != {r["sample_id"] for r in v11}:
        raise SystemExit("v1/v1_1 sample_id sets differ")
    if any(v1_by[r["sample_id"]]["split"] != r["split"] for r in v11):
        raise SystemExit("split assignments changed; refuse auto-resplit")

    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for r in v11:
        by_split[r["split"]][r["final_label"] or "empty"] += 1
    single = [
        s
        for s, c in by_split.items()
        if sum(1 for k in ("0", "1") if c.get(k, 0) > 0) < 2
    ]
    joined = []
    for r in v11:
        o = v1_by[r["sample_id"]]
        joined.append(
            {
                "split": r["split"],
                "sha256": o.get("sha256") or r.get("raw_sha256"),
                "duplicate_group_id": r.get("duplicate_group_id") or o.get("duplicate_group_id"),
                "scene_group_id": r.get("scene_group_id") or o.get("scene_group_id"),
                "source_id": o.get("source_id"),
                "sequence_id": o.get("sequence_id"),
            }
        )
    leakage = check_group_leakage(joined, GROUP_FIELDS)
    return {
        "splits_unchanged": True,
        "class_distribution": {k: dict(v) for k, v in sorted(by_split.items())},
        "single_class_splits": single,
        "leakage_errors": leakage,
        "split_v2_required": bool(single or leakage),
    }


def remetrics(*, pred_rows: list[dict[str, str]], v11: list[dict[str, str]]) -> dict[str, Any]:
    if any(str(r.get("split") or "").lower() == "test" for r in pred_rows):
        raise SystemExit("refuse: test-split rows in frozen validation predictions")
    labels = {
        r["sample_id"]: r["final_label"]
        for r in v11
        if str(r.get("split") or "").lower() == "validation"
    }
    y_list: list[int] = []
    s_list: list[float] = []
    for row in pred_rows:
        sid = row["sample_id"]
        if sid not in labels:
            raise SystemExit(f"validation prediction sample missing from v1_1: {sid}")
        lab = labels[sid]
        if lab not in {"0", "1"}:
            raise SystemExit(f"non-binary v1_1 validation label for {sid}: {lab!r}")
        y_list.append(int(lab))
        s_list.append(float(row["anomaly_score"]))
    if len(y_list) != 54:
        raise SystemExit(f"expected 54 validation predictions, got {len(y_list)}")
    y = np.asarray(y_list, dtype=np.int32)
    s = orient_scores(np.asarray(s_list, dtype=np.float64), higher_means_anomalous=True)
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        raise SystemExit("validation single-class under v1_1; refuse threshold selection")
    threshold, f1_at = select_threshold_on_validation(
        y, s, metric="f1", tie_break="highest_threshold"
    )
    if threshold is None:
        raise SystemExit("threshold selection failed on v1_1 validation")
    y_pred = (s >= float(threshold)).astype(np.int32)
    cm = confusion(y, y_pred)
    f1, precision, recall = f1_precision_recall(cm)
    return {
        "config_id": FROZEN_CONFIG_ID,
        "precision_mode": "fp32",
        "inference_rerun": False,
        "scores_source": str(FROZEN_PRED.relative_to(REPO_ROOT)).replace("\\", "/"),
        "label_source": "independent_eval_v1_1",
        "validation_n": int(y.size),
        "validation_positive_count": int((y == 1).sum()),
        "validation_negative_count": int((y == 0).sum()),
        "auroc": binary_auroc(y, s),
        "average_precision": average_precision(y, s),
        "selected_threshold": float(threshold),
        "validation_f1_at_threshold": f1_at,
        "F1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "profile_selection_mutated": False,
        "test_inference_performed": False,
    }


def v1_frozen_metrics_snapshot() -> dict[str, Any]:
    sel = json.loads(PROFILE_SELECTION.read_text(encoding="utf-8"))
    for cand in sel.get("candidates") or []:
        if cand.get("config_id") == FROZEN_CONFIG_ID:
            return dict(cand["metrics"])
    raise SystemExit(f"{FROZEN_CONFIG_ID} missing from profile_selection.json")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    if not V11_MANIFEST.is_file():
        raise SystemExit("pending_v1_1_freeze: independent_eval_v1_1.csv missing")
    v1 = _read_csv(V1_MANIFEST)
    v11 = _read_csv(V11_MANIFEST)
    pred = _read_csv(FROZEN_PRED)
    integrity = split_integrity(v11, v1)
    if integrity["split_v2_required"]:
        raise SystemExit(
            "STOP: propose independent_eval_v1_1_split_v2 "
            f"(single_class={integrity['single_class_splits']} leakage={integrity['leakage_errors']})"
        )
    metrics = remetrics(pred_rows=pred, v11=v11)
    payload = {
        "annotation_version": "independent_eval_v1_1",
        "evaluation_name": "independent_eval_v1_1 evaluation",
        "not_a_historical_paper_correction": True,
        "split_integrity": integrity,
        "v1_heuristic_label_metrics": v1_frozen_metrics_snapshot(),
        "v1_1_human_reviewed_metrics": metrics,
        "final_test_recommendation": "keep_closed",
        "test_opened": False,
        "final_test_authorized": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "v1_1_validation_relabel_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {
            "v1_1_auroc": metrics["auroc"],
            "v1_1_ap": metrics["average_precision"],
            "v1_1_f1": metrics["F1"],
            "v1_1_threshold": metrics["selected_threshold"],
            "v1_1_cm": metrics["confusion_matrix"],
            "class_distribution": integrity["class_distribution"],
            "final_test_recommendation": "keep_closed",
        }
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
