"""Legacy PaDiM/PatchCore exploratory baseline scoring (validation only)."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common import REPO_ROOT, read_csv_dicts, resolve_repo_path, resolve_under_dataset_root, sha256_file, write_json, write_text  # noqa: E402
from frozen_checkpoint_registry import _resolve_path, load_registry, verify_registry_entry  # noqa: E402
from test_split_embargo import assert_split_allowed  # noqa: E402

REQUIRED_KEYS = (
    "baseline_type",
    "checkpoint_path",
    "checkpoint_sha256",
    "backbone_weights_sha256",
    "score_aggregation",
    "dataset_manifest",
    "dataset_root_env",
    "validation_split",
    "model_name",
    "model_version",
    "config_id",
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be mapping: {path}")
    return data


def _require_contract(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [k for k in REQUIRED_KEYS if not config.get(k)]
    if missing:
        errors.append(f"missing contract keys: {missing}")
    if config.get("evaluation_role") != "secondary_exploratory":
        errors.append("evaluation_role must be secondary_exploratory")
    if config.get("score_aggregation") != "max_anomaly_map_exploratory":
        errors.append("score_aggregation must be max_anomaly_map_exploratory")
    return errors


def _verify_backbone_sha(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = str(config.get("backbone_weights_sha256", "")).lower()
    if not expected:
        return ["backbone_weights_sha256 missing"]
    reg = load_registry()
    entry = next(
        (e for e in reg.get("checkpoints") or [] if e.get("checkpoint_id") == "torchvision_wide_resnet50_2"),
        None,
    )
    if entry is None:
        errors.append("registry missing torchvision_wide_resnet50_2 entry")
        return errors
    path = _resolve_path(entry)
    if path is None or not path.is_file():
        errors.append(f"backbone weights missing offline: {entry.get('local_cache_path_windows')}")
        return errors
    actual = sha256_file(path).lower()
    if actual != expected:
        errors.append(f"backbone sha mismatch expected={expected} actual={actual}")
    return errors


def _image_score_from_map(amap: np.ndarray) -> float:
    return float(np.max(amap))


def _run_padim(config: dict[str, Any], image_path: Path) -> float:
    from src.models.anomaly.padim import PaDiM, PaDiMConfig

    ckpt_path = resolve_repo_path(str(config["checkpoint_path"]))
    model = PaDiM(PaDiMConfig(device="cpu"))
    model.load(str(ckpt_path))
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    amap = model.predict_anomaly_map(rgb)
    return _image_score_from_map(amap)


def _run_patchcore(config: dict[str, Any], image_path: Path) -> float:
    from src.models.anomaly.patchcore import PatchCore, PatchCoreConfig

    ckpt_path = resolve_repo_path(str(config["checkpoint_path"]))
    model = PatchCore(PatchCoreConfig(device="cpu"))
    model.load(str(ckpt_path))
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    amap = model.predict_anomaly_map(rgb)
    return _image_score_from_map(amap)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run legacy exploratory baseline on validation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default="legacy_exploratory")
    args = ap.parse_args(argv)

    config_path = resolve_repo_path(str(args.config))
    config = _load_config(config_path)
    contract_errors = _require_contract(config)
    if contract_errors:
        for err in contract_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    ckpt_path = resolve_repo_path(str(config["checkpoint_path"]))
    expected_sha = str(config["checkpoint_sha256"]).lower()
    if not ckpt_path.is_file():
        print(f"ERROR: checkpoint missing: {ckpt_path}", file=sys.stderr)
        return 2
    if sha256_file(ckpt_path).lower() != expected_sha:
        print("ERROR: checkpoint sha256 mismatch", file=sys.stderr)
        return 2

    reg = load_registry()
    checkpoint_id = "padim_stats_legacy" if config["baseline_type"] == "padim" else "patchcore_bank_legacy"
    entry = next((e for e in reg.get("checkpoints") or [] if e.get("checkpoint_id") == checkpoint_id), None)
    if entry is None:
        print(f"ERROR: registry entry missing for {checkpoint_id}", file=sys.stderr)
        return 2
    reg_errors = verify_registry_entry(entry, load_models=False)
    backbone_errors = _verify_backbone_sha(config)
    if reg_errors or backbone_errors:
        for err in reg_errors + backbone_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    env_key = str(config.get("dataset_root_env", "ARTPS_DATASET_ROOT"))
    dataset_root_raw = os.environ.get(env_key)
    if not dataset_root_raw:
        print(f"ERROR: {env_key} is not set", file=sys.stderr)
        return 2
    dataset_root = Path(dataset_root_raw).resolve()

    manifest_path = resolve_repo_path(str(config["dataset_manifest"]))
    rows = read_csv_dicts(manifest_path)
    val_split = str(config.get("validation_split", "validation"))
    assert_split_allowed(val_split)
    if val_split == "test":
        print("ERROR: test split refused", file=sys.stderr)
        return 2

    scorer = _run_padim if config["baseline_type"] == "padim" else _run_patchcore
    out_dir = resolve_repo_path(str(config.get("output_directory", "results/iac2026/independent_eval_v1/validation")))
    run_dir = out_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = run_dir / "predictions.csv"

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split") != val_split:
            continue
        image_path = resolve_under_dataset_root(dataset_root, str(row["relative_path"]))
        score = scorer(config, image_path)
        csv_rows.append(
            {
                "sample_id": row["sample_id"],
                "split": val_split,
                "y_true": int(row["binary_label"]),
                "anomaly_score": score,
                "model_name": config["model_name"],
                "model_version": config["model_version"],
                "config_id": config["config_id"],
            }
        )

    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "split",
                "y_true",
                "anomaly_score",
                "model_name",
                "model_version",
                "config_id",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    write_json(
        run_dir / "provenance.json",
        {
            "evaluation_role": config.get("evaluation_role"),
            "score_aggregation": config.get("score_aggregation"),
            "not_final_test_result": bool(config.get("not_final_test_result", True)),
            "eligible_for_manuscript_primary_results": False,
            "eligible_for_primary_baseline_table": False,
            "eligible_for_C06_reproduction": False,
            "checkpoint_sha256": expected_sha,
            "predictions_sha256": sha256_file(pred_csv),
        },
    )
    write_text(run_dir / "command.txt", " ".join(sys.argv) + "\n")
    print(f"OK: wrote {len(csv_rows)} exploratory baseline predictions to {pred_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
