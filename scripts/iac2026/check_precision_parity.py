"""Compare FP32 vs AMP frozen ARTPS predictions (fail-closed parity gates)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, write_json  # noqa: E402
from detection_metrics_lib import average_precision, binary_auroc  # noqa: E402

SCORE_ABS_TOL = 1e-4
METRIC_ABS_TOL = 1e-4


def _read_predictions(path: Path) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["sample_id"], row["split"])
            out[key] = float(row["anomaly_score"])
    return out


def _load_split_scores(path: Path, split: str) -> tuple[list[int], list[float]]:
    y: list[int] = []
    s: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != split:
                continue
            y.append(int(row["y_true"]))
            s.append(float(row["anomaly_score"]))
    return y, s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check FP32 vs AMP precision parity")
    ap.add_argument("--fp32", required=True, help="FP32 reference predictions CSV")
    ap.add_argument("--amp", required=True, help="AMP candidate predictions CSV")
    ap.add_argument(
        "--output",
        default="results/iac2026/independent_eval_v1/validation/precision_parity.json",
    )
    ap.add_argument("--split", default="validation")
    args = ap.parse_args(argv)

    fp32_path = (REPO_ROOT / args.fp32).resolve() if not Path(args.fp32).is_absolute() else Path(args.fp32)
    amp_path = (REPO_ROOT / args.amp).resolve() if not Path(args.amp).is_absolute() else Path(args.amp)

    if not fp32_path.is_file():
        print(f"ERROR: FP32 reference missing: {fp32_path}", file=sys.stderr)
        return 2
    if not amp_path.is_file():
        print(f"ERROR: AMP predictions missing: {amp_path}", file=sys.stderr)
        return 2

    fp32 = _read_predictions(fp32_path)
    amp = _read_predictions(amp_path)
    errors: list[str] = []
    max_abs = 0.0
    compared = 0
    for key, ref_score in fp32.items():
        if key[1] != args.split:
            continue
        if key not in amp:
            errors.append(f"missing AMP row for {key}")
            continue
        diff = abs(ref_score - amp[key])
        max_abs = max(max_abs, diff)
        compared += 1
        if diff > SCORE_ABS_TOL:
            errors.append(f"score parity fail {key}: diff={diff:.6g} > {SCORE_ABS_TOL}")

    y_fp32, s_fp32 = _load_split_scores(fp32_path, args.split)
    y_amp, s_amp = _load_split_scores(amp_path, args.split)
    y_fp32_arr = np.asarray(y_fp32, dtype=np.int32)
    s_fp32_arr = np.asarray(s_fp32, dtype=np.float64)
    s_amp_arr = np.asarray(s_amp, dtype=np.float64)
    ap_fp32 = average_precision(y_fp32_arr, s_fp32_arr)
    ap_amp = average_precision(y_fp32_arr, s_amp_arr)
    auc_fp32 = binary_auroc(y_fp32_arr, s_fp32_arr)
    auc_amp = binary_auroc(y_fp32_arr, s_amp_arr)
    if ap_fp32 is not None and ap_amp is not None and abs(ap_fp32 - ap_amp) > METRIC_ABS_TOL:
        errors.append(f"AP parity fail: fp32={ap_fp32} amp={ap_amp}")
    if auc_fp32 is not None and auc_amp is not None and abs(auc_fp32 - auc_amp) > METRIC_ABS_TOL:
        errors.append(f"AUROC parity fail: fp32={auc_fp32} amp={auc_amp}")

    passed = not errors
    report = {
        "passed": passed,
        "split": args.split,
        "gates": {
            "score_abs_tolerance": SCORE_ABS_TOL,
            "metric_abs_tolerance": METRIC_ABS_TOL,
        },
        "compared_rows": compared,
        "max_abs_score_diff": max_abs,
        "fp32_predictions_csv": str(fp32_path),
        "amp_predictions_csv": str(amp_path),
        "validation_average_precision_fp32": ap_fp32,
        "validation_average_precision_amp": ap_amp,
        "validation_auroc_fp32": auc_fp32,
        "validation_auroc_amp": auc_amp,
        "errors": errors,
    }
    out_path = (REPO_ROOT / args.output).resolve()
    write_json(out_path, report)
    if not passed:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2
    print(f"OK: precision parity passed ({compared} rows, max_diff={max_abs:.3g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
