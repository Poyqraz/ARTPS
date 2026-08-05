"""Select frozen ARTPS validation profile from predeclared profile prediction CSVs."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, write_json  # noqa: E402
from detection_metrics_lib import (  # noqa: E402
    average_precision,
    binary_auroc,
    orient_scores,
    select_threshold_on_validation,
)

DEFAULT_PROFILES = (
    ("artps_full_frozen_raw_clf_on_v1", "results/iac2026/independent_eval_v1/validation/artps_full_frozen_raw_clf_on_v1/predictions.csv"),
    ("artps_full_frozen_raw_clf_off_v1", "results/iac2026/independent_eval_v1/validation/artps_full_frozen_raw_clf_off_v1/predictions.csv"),
    ("artps_full_frozen_mars_clf_on_v1", "results/iac2026/independent_eval_v1/validation/artps_full_frozen_mars_clf_on_v1/predictions.csv"),
    ("artps_full_frozen_mars_clf_off_v1", "results/iac2026/independent_eval_v1/validation/artps_full_frozen_mars_clf_off_v1/predictions.csv"),
)

PROFILE_META = {
    "artps_full_frozen_raw_clf_on_v1": {"classifier_off": False, "raw_rgb": True},
    "artps_full_frozen_raw_clf_off_v1": {"classifier_off": True, "raw_rgb": True},
    "artps_full_frozen_mars_clf_on_v1": {"classifier_off": False, "raw_rgb": False},
    "artps_full_frozen_mars_clf_off_v1": {"classifier_off": True, "raw_rgb": False},
}


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _validation_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    y: list[int] = []
    s: list[float] = []
    for row in rows:
        if row.get("split") != "validation":
            continue
        y.append(int(row["y_true"]))
        s.append(float(row["anomaly_score"]))
    return np.asarray(y, dtype=np.int32), np.asarray(s, dtype=np.float64)


def _profile_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    y, s_raw = _validation_arrays(rows)
    s = orient_scores(s_raw, higher_means_anomalous=True)
    ap = average_precision(y, s)
    auroc = binary_auroc(y, s)
    threshold, f1_at_threshold = select_threshold_on_validation(
        y, s, metric="f1", tie_break="highest_threshold"
    )
    return {
        "validation_n": int(y.size),
        "average_precision": ap,
        "auroc": auroc,
        "selected_threshold": threshold,
        "validation_f1_at_threshold": f1_at_threshold,
    }


def _better(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return True if profile a beats b under selection policy."""
    ap_a = a["metrics"].get("average_precision")
    ap_b = b["metrics"].get("average_precision")
    if ap_a is None and ap_b is None:
        pass
    elif ap_a is None:
        return False
    elif ap_b is None:
        return True
    elif ap_a > ap_b:
        return True
    elif ap_a < ap_b:
        return False

    auc_a = a["metrics"].get("auroc")
    auc_b = b["metrics"].get("auroc")
    if auc_a is not None and auc_b is not None:
        if auc_a > auc_b:
            return True
        if auc_a < auc_b:
            return False

    meta_a = PROFILE_META.get(a["config_id"], {})
    meta_b = PROFILE_META.get(b["config_id"], {})
    if meta_a.get("classifier_off") and not meta_b.get("classifier_off"):
        return True
    if meta_b.get("classifier_off") and not meta_a.get("classifier_off"):
        return False

    if meta_a.get("raw_rgb") and not meta_b.get("raw_rgb"):
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select frozen ARTPS validation profile")
    ap.add_argument(
        "--predictions",
        action="append",
        default=[],
        help="config_id=path/to/predictions.csv (repeatable)",
    )
    ap.add_argument(
        "--output",
        default="results/iac2026/independent_eval_v1/validation/profile_selection.json",
    )
    args = ap.parse_args(argv)

    entries: list[tuple[str, Path]] = []
    if args.predictions:
        for item in args.predictions:
            if "=" not in item:
                print(f"ERROR: expected config_id=path, got {item!r}", file=sys.stderr)
                return 2
            cid, raw = item.split("=", 1)
            entries.append((cid.strip(), (REPO_ROOT / raw).resolve()))
    else:
        for cid, rel in DEFAULT_PROFILES:
            entries.append((cid, (REPO_ROOT / rel).resolve()))

    profiles: list[dict[str, Any]] = []
    for config_id, path in entries:
        if not path.is_file():
            print(f"ERROR: missing predictions for {config_id}: {path}", file=sys.stderr)
            return 2
        rows = _read_predictions(path)
        metrics = _profile_metrics(rows)
        profiles.append(
            {
                "config_id": config_id,
                "predictions_csv": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "metrics": metrics,
            }
        )

    best = profiles[0]
    for candidate in profiles[1:]:
        if _better(candidate, best):
            best = candidate

    artifact = {
        "protocol_id": "independent_eval_v1",
        "selection_split": "validation",
        "selection_policy": {
            "primary_metric": "average_precision",
            "tie_breakers": ["auroc", "classifier_off", "raw_rgb_v1"],
            "threshold_metric": "f1",
            "threshold_tie_break": "highest_threshold",
        },
        "not_final_test_result": True,
        "eligible_for_manuscript_primary_results": False,
        "selected_config_id": best["config_id"],
        "selected_predictions_csv": best["predictions_csv"],
        "selected_metrics": best["metrics"],
        "candidates": profiles,
    }

    out_path = (REPO_ROOT / args.output).resolve()
    write_json(out_path, artifact)
    print(f"OK: selected {best['config_id']} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
