"""Run frozen ARTPS full-profile inference for independent_eval_v1 validation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common import (  # noqa: E402
    REPO_ROOT,
    git_dirty,
    git_head,
    load_json_schema,
    make_run_id,
    read_csv_dicts,
    resolve_repo_path,
    resolve_under_dataset_root,
    sha256_file,
    validate_rows,
    write_json,
    write_text,
)
from artps_full_profile_cache import (  # noqa: E402
    allowed_splits,
    build_cache_index,
    build_metrics_config_snapshot,
    cache_dir_for_profile,
    environment_snapshot_torch,
    load_profile_yaml,
    profile_to_frozen_kwargs,
    verify_profile_registry,
)
from src.artps_inference import (  # noqa: E402
    FrozenARTPSConfig,
    load_frozen_artps_profile,
    predict_image,
)
from test_split_embargo import assert_split_allowed  # noqa: E402


PREDICTION_COLUMNS = (
    "sample_id",
    "split",
    "y_true",
    "anomaly_score",
    "model_name",
    "model_version",
    "config_id",
)


def _dataset_root(profile: dict[str, Any]) -> Path:
    env_key = str(profile.get("dataset_root_env", "ARTPS_DATASET_ROOT"))
    raw = os.environ.get(env_key)
    if not raw:
        raise RuntimeError(f"{env_key} is not set")
    return Path(raw).resolve()


def _write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PREDICTION_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in PREDICTION_COLUMNS})


def _filter_manifest_rows(profile: dict[str, Any], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    splits = set(allowed_splits(profile))
    out: list[dict[str, str]] = []
    for row in rows:
        split = str(row.get("split", ""))
        if split not in splits:
            continue
        assert_split_allowed(split)
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run frozen ARTPS full-profile batch inference")
    ap.add_argument("--profile", required=True, help="Profile YAML path")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--split", default=None, help="Optional single split override")
    ap.add_argument("--cache-only", action="store_true", help="Allow train split for cache generation")
    args = ap.parse_args(argv)

    profile = load_profile_yaml(args.profile)
    reg_errors = verify_profile_registry(profile)
    if reg_errors:
        for err in reg_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    if args.split:
        assert_split_allowed(args.split)
        if str(args.split).lower() == "test":
            print("ERROR: test split refused by runner", file=sys.stderr)
            return 2

    manifest_path = resolve_repo_path(str(profile["dataset_manifest"]))
    manifest_rows = read_csv_dicts(manifest_path)
    if args.split:
        manifest_rows = [r for r in manifest_rows if r.get("split") == args.split]
    else:
        manifest_rows = _filter_manifest_rows(profile, manifest_rows)
        if not args.cache_only:
            manifest_rows = [r for r in manifest_rows if r.get("split") != "train"]

    if not manifest_rows:
        print("ERROR: no manifest rows for requested splits", file=sys.stderr)
        return 2

    run_id = args.run_id or make_run_id(str(profile["config_id"]))
    out_dir = resolve_repo_path(str(profile.get("output_directory", "results/iac2026/independent_eval_v1/validation")))
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    profile_path = resolve_repo_path(str(args.profile))
    snapshot_path = run_dir / "run_config.snapshot.yaml"
    snapshot_path.write_bytes(profile_path.read_bytes())

    frozen_cfg = FrozenARTPSConfig(**profile_to_frozen_kwargs(profile))
    bundle = load_frozen_artps_profile(frozen_cfg)
    dataset_root = _dataset_root(profile)

    csv_rows: list[dict[str, Any]] = []
    jsonl_path = run_dir / "predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in manifest_rows:
            split = str(row["split"])
            assert_split_allowed(split)
            sample_id = str(row["sample_id"])
            image_path = resolve_under_dataset_root(dataset_root, str(row["relative_path"]))
            record = predict_image(
                image_path,
                bundle,
                frozen_cfg,
                sample_id=sample_id,
                split=split,
            )
            jf.write(json.dumps(record, allow_nan=False) + "\n")
            csv_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "y_true": int(row["binary_label"]),
                    "anomaly_score": float(record.get("image_score", 0.0)),
                    "model_name": str(profile.get("model_name", bundle.model_name)),
                    "model_version": str(profile.get("model_version", bundle.model_version)),
                    "config_id": str(profile["config_id"]),
                }
            )

    pred_csv = run_dir / "predictions.csv"
    if profile.get("predictions_csv"):
        pred_csv = resolve_repo_path(str(profile["predictions_csv"]))
    _write_predictions_csv(pred_csv, csv_rows)

    schema = load_json_schema("prediction_table.schema.json")
    schema_errors = validate_rows(csv_rows, schema, coerce_ints=("y_true",), coerce_floats=("anomaly_score",))
    if schema_errors:
        print("PREDICTION SCHEMA FAILED:\n- " + "\n- ".join(schema_errors), file=sys.stderr)
        return 2

    metrics_cfg = build_metrics_config_snapshot(profile, predictions_csv=pred_csv, output_directory=run_dir)
    metrics_yaml_path = run_dir / "metrics_config.snapshot.yaml"
    metrics_yaml_path.write_text(
        yaml.safe_dump(metrics_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    registry_path = resolve_repo_path(
        str(profile.get("registry_path", "reproduction/iac2026/frozen_checkpoint_registry.yaml"))
    )
    shutil.copy2(registry_path, run_dir / "checkpoint_registry.snapshot.yaml")

    write_json(run_dir / "environment.json", environment_snapshot_torch())
    write_json(
        run_dir / "input_audit.json",
        {
            "passed": True,
            "evidence_mode": "real_evidence",
            "claim_ids": list(profile.get("claim_ids") or ["IND_EVAL_V1"]),
            "config_id": str(profile["config_id"]),
            "config_sha256": sha256_file(profile_path),
            "manifest_sha256": sha256_file(manifest_path),
            "predictions_sha256": sha256_file(pred_csv),
            "checkpoint_sha256": (profile.get("autoencoder") or {}).get("checkpoint_sha256"),
            "git_head": git_head(),
            "git_dirty": git_dirty(),
            "protocol_id": profile.get("protocol_id"),
            "protocol_lock_sha256": profile.get("protocol_lock_sha256"),
            "evaluation_purpose": profile.get("evaluation_purpose"),
            "annotation_version": profile.get("annotation_version"),
            "image_score_aggregation": profile.get("image_score_aggregation"),
            "blockers": [],
            "errors": [],
        },
    )
    write_json(
        run_dir / "provenance.json",
        {
            "git_head": git_head(),
            "git_dirty": git_dirty(),
            "config_sha256": sha256_file(profile_path),
            "manifest_sha256": sha256_file(manifest_path),
            "predictions_sha256": sha256_file(pred_csv),
            "profile_id": profile.get("profile_id"),
            "not_final_test_result": bool(profile.get("not_final_test_result", True)),
            "eligible_for_manuscript_primary_results": bool(
                profile.get("eligible_for_manuscript_primary_results", False)
            ),
        },
    )
    cmd = " ".join(sys.argv)
    write_text(run_dir / "command.txt", cmd + "\n")

    cache_dir = cache_dir_for_profile(profile)
    cache_index = build_cache_index(cache_dir) if cache_dir.is_dir() else {"cache_dir": str(cache_dir), "sample_count": 0}
    write_json(run_dir / "cache_index.json", cache_index)

    print(f"OK: wrote {len(csv_rows)} predictions to {pred_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
