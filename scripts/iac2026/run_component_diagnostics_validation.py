"""Instrumented validation-only component diagnostics for frozen ARTPS profile.

Does not open the test split. Does not rewrite predictions.csv.
Writes component_diagnostics_v1.csv and asserts image_score parity with committed predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "iac2026"))
sys.path.insert(0, str(REPO_ROOT))

from artps_full_profile_cache import (  # noqa: E402
    load_profile_yaml,
    profile_to_frozen_kwargs,
    verify_profile_registry,
)
from test_split_embargo import assert_final_test_authorized, assert_split_allowed  # noqa: E402

DEFAULT_PROFILE = (
    REPO_ROOT
    / "reproduction"
    / "iac2026"
    / "configs"
    / "independent_eval_artps_full_frozen_mars.yaml"
)
COMMITTED_PRED = (
    REPO_ROOT
    / "results"
    / "iac2026"
    / "independent_eval_v1"
    / "validation"
    / "artps_full_frozen_mars_clf_on_v1"
    / "predictions.csv"
)
OUT_CSV = (
    REPO_ROOT
    / "results"
    / "iac2026"
    / "independent_eval_v1"
    / "validation"
    / "artps_full_frozen_mars_clf_on_v1"
    / "component_diagnostics_v1.csv"
)
MANIFEST = REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1.csv"

COMPONENT_FIELDS = [
    "sample_id",
    "y_true",
    "image_score",
    "raw_proposal_count",
    "scored_candidate_count",
    "kept_candidate_count",
    "suppressed_candidate_count",
    "top_candidate_box",
    "combined_pool",
    "depth_pool",
    "detector_confidence",
    "classifier_argmax",
    "classifier_logits_or_probabilities",
    "classifier_known_value",
    "padim_pool",
    "patchcore_pool",
    "local_value",
    "anomaly_score_before_gate",
    "final_candidate_score",
    "keep_or_drop",
    "drop_reason",
    "mask_reason",
    "no_valid_candidate_reason",
    "execution_path",
    "warning_flags",
    "score_parity_ok",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--committed-predictions", type=Path, default=COMMITTED_PRED)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args(argv)

    if str(args.split).strip().lower() != "validation":
        raise SystemExit("refusing non-validation split (final test closed)")
    assert_split_allowed("validation")
    # Explicitly refuse if somehow authorized incorrectly while blocked.
    try:
        assert_final_test_authorized()
        raise SystemExit("refusing: final_test_authorized unexpectedly true")
    except ValueError:
        pass

    root = (os.environ.get("ARTPS_DATASET_ROOT") or "").strip()
    if not root:
        raise SystemExit("ARTPS_DATASET_ROOT unset (fail closed)")
    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise SystemExit(f"ARTPS_DATASET_ROOT not a directory: {dataset_root}")

    pred_sha_before = _sha256(args.committed_predictions)
    committed = {r["sample_id"]: r for r in _read_csv(args.committed_predictions)}

    from src.artps_inference import FrozenARTPSConfig, load_frozen_artps_profile, predict_image

    profile = load_profile_yaml(args.profile)
    errs = verify_profile_registry(profile)
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 2

    kwargs = profile_to_frozen_kwargs(profile)
    cfg = FrozenARTPSConfig(**kwargs)
    bundle = load_frozen_artps_profile(cfg)

    manifest = [
        r
        for r in _read_csv(MANIFEST)
        if str(r.get("split")).strip().lower() == "validation"
        and str(r.get("inclusion_status")).strip().lower() == "included"
        and str(r.get("adjudication_status")).strip().lower() == "resolved"
    ]
    if len(manifest) != 54:
        raise SystemExit(f"expected 54 validation rows, got {len(manifest)}")

    out_rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for row in manifest:
        sid = row["sample_id"]
        if sid not in committed:
            raise SystemExit(f"missing committed prediction for {sid}")
        img = dataset_root / row["relative_path"]
        if not img.is_file():
            raise SystemExit(f"missing image: {img}")
        diag: dict[str, Any] = {}
        rec = predict_image(
            img,
            bundle,
            cfg,
            sample_id=sid,
            split="validation",
            diagnostics_out=diag,
        )
        committed_score = float(committed[sid]["anomaly_score"])
        got = float(rec["image_score"])
        parity = abs(got - committed_score) <= args.tol
        if not parity:
            mismatches.append(f"{sid}: committed={committed_score} got={got}")
        if rec.get("processing_status") == "error":
            raise SystemExit(f"processing_status=error for {sid}: {rec.get('warning_flags')}")

        out_rows.append(
            {
                "sample_id": sid,
                "y_true": committed[sid]["y_true"],
                "image_score": got,
                "raw_proposal_count": diag.get("raw_proposal_count", ""),
                "scored_candidate_count": diag.get("scored_candidate_count", ""),
                "kept_candidate_count": diag.get("kept_candidate_count", ""),
                "suppressed_candidate_count": diag.get("suppressed_candidate_count", ""),
                "top_candidate_box": diag.get("top_candidate_box", ""),
                "combined_pool": diag.get("combined_pool", ""),
                "depth_pool": diag.get("depth_pool", ""),
                "detector_confidence": diag.get("detector_confidence", ""),
                "classifier_argmax": diag.get("classifier_argmax", ""),
                "classifier_logits_or_probabilities": diag.get(
                    "classifier_logits_or_probabilities", ""
                ),
                "classifier_known_value": diag.get("classifier_known_value", ""),
                "padim_pool": diag.get("padim_map", diag.get("padim_pool", "")),
                "patchcore_pool": diag.get("patchcore_pool", ""),
                "local_value": diag.get("local_value", ""),
                "anomaly_score_before_gate": diag.get("anomaly_score_before_gate", ""),
                "final_candidate_score": diag.get("final_candidate_score", ""),
                "keep_or_drop": diag.get("keep_or_drop", ""),
                "drop_reason": diag.get("drop_reason", ""),
                "mask_reason": diag.get("mask_reason", ""),
                "no_valid_candidate_reason": diag.get("no_valid_candidate_reason", ""),
                "execution_path": diag.get("execution_path", ""),
                "warning_flags": diag.get("warning_flags", ""),
                "score_parity_ok": parity,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPONENT_FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    pred_sha_after = _sha256(args.committed_predictions)
    if pred_sha_before != pred_sha_after:
        raise SystemExit("predictions.csv SHA changed during diagnostics run — refuse")

    summary = {
        "n": len(out_rows),
        "score_parity_mismatches": mismatches,
        "predictions_csv_sha256": pred_sha_after,
        "out_csv": str(args.out_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
        "final_test_authorized": False,
        "split": "validation",
    }
    summary_path = args.out_csv.with_name("component_diagnostics_v1_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if mismatches:
        print("SCORE PARITY FAILED:", file=sys.stderr)
        for m in mismatches[:20]:
            print(m, file=sys.stderr)
        return 3

    print(f"Wrote {args.out_csv} n={len(out_rows)} parity_ok predictions_sha={pred_sha_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
