"""Fail-closed audit of IAC reproduction inputs (manifest / predictions / git)."""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    REPO_ROOT,
    copy_config_sidecar,
    environment_snapshot,
    git_dirty,
    load_json_schema,
    load_yaml,
    make_run_id,
    read_csv_dicts,
    resolve_repo_path,
    sha256_file,
    validate_rows,
    relpath_or_abs,
    write_json,
    write_text,
)

BLOCKER_TASK_LEVEL = "TASK_LEVEL_TBD"


def _audit_manifest(rows: List[Dict[str, str]], *, require_real_sha256: bool) -> List[str]:
    errors: List[str] = []
    schema = load_json_schema("dataset_manifest.schema.json")
    errors.extend(
        validate_rows(
            rows,
            schema,
            coerce_ints=["binary_label"],
        )
    )
    ids = [r.get("sample_id", "") for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample_id in manifest")

    # Scene / duplicate leakage across splits
    scene_splits: Dict[str, Set[str]] = defaultdict(set)
    dup_splits: Dict[str, Set[str]] = defaultdict(set)
    for r in rows:
        scene_splits[r.get("scene_group_id", "")].add(r.get("split", ""))
        dup_splits[r.get("duplicate_group_id", "")].add(r.get("split", ""))
    for scene, splits in scene_splits.items():
        if scene and len(splits) > 1:
            errors.append(f"scene_group_id {scene!r} spans splits {sorted(splits)}")
    for dup, splits in dup_splits.items():
        if dup and len(splits) > 1:
            errors.append(f"duplicate_group_id {dup!r} spans splits {sorted(splits)}")

    if require_real_sha256:
        for i, r in enumerate(rows):
            digest = r.get("sha256", "")
            if digest.startswith("SYNTHETIC_"):
                errors.append(f"row {i}: synthetic sha256 not allowed when require_real_sha256")
            elif len(digest) != 64:
                errors.append(f"row {i}: sha256 must be 64 hex chars")
    return errors


def _audit_predictions(rows: List[Dict[str, str]]) -> List[str]:
    errors: List[str] = []
    schema = load_json_schema("prediction_table.schema.json")
    errors.extend(
        validate_rows(
            rows,
            schema,
            coerce_ints=["y_true"],
            coerce_floats=["anomaly_score"],
        )
    )
    for i, r in enumerate(rows):
        try:
            score = float(r["anomaly_score"])
        except (KeyError, TypeError, ValueError):
            continue
        if score != score:  # NaN
            errors.append(f"row {i}: anomaly_score is NaN")
    labels = {int(r["y_true"]) for r in rows if r.get("y_true", "") != ""}
    if labels and labels <= {0}:
        errors.append("predictions contain only negative class")
    if labels and labels <= {1}:
        errors.append("predictions contain only positive class")
    return errors


def _balance_report(rows: List[Dict[str, str]], label_key: str, split_key: str = "split") -> Dict[str, Any]:
    by_split: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        split = r.get(split_key, "")
        try:
            label = int(r[label_key])
        except (KeyError, ValueError, TypeError):
            continue
        by_split[split][label] += 1
    return {s: dict(c) for s, c in sorted(by_split.items())}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit IAC 2026 reproduction inputs (fail closed).")
    ap.add_argument("--config", required=True, help="Path to detection reproduction YAML")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    config_path = resolve_repo_path(args.config)
    cfg = load_yaml(config_path)
    run_id = args.run_id or make_run_id("audit")
    out_dir = resolve_repo_path(cfg.get("output_directory", "results/iac2026/reproduction")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: List[str] = []
    blockers: List[str] = []
    warnings: List[str] = []

    task_level = str(cfg.get("task_level", BLOCKER_TASK_LEVEL))
    if task_level == BLOCKER_TASK_LEVEL or not task_level or task_level.upper().endswith("_TBD"):
        blockers.append("task_level is TBD / unknown — real C05/C06 inference blocked")

    threshold_policy = str(cfg.get("threshold_policy", "unknown")).lower()
    if threshold_policy not in {"fixed_historical", "validation_selected"}:
        blockers.append(
            f"threshold_policy={threshold_policy!r} — must be fixed_historical or validation_selected"
        )

    allow_dirty = bool(cfg.get("allow_dirty_git", False))
    dirty = git_dirty()
    if dirty and not allow_dirty:
        errors.append("git working tree is dirty (set allow_dirty_git: true only for local dry-runs)")

    manifest_path = resolve_repo_path(str(cfg["dataset_manifest"]))
    pred_path = resolve_repo_path(str(cfg["predictions_csv"]))
    if not manifest_path.is_file():
        errors.append(f"manifest missing: {manifest_path}")
    if not pred_path.is_file():
        errors.append(f"predictions missing: {pred_path}")

    manifest_rows: List[Dict[str, str]] = []
    pred_rows: List[Dict[str, str]] = []
    if manifest_path.is_file():
        manifest_rows = read_csv_dicts(manifest_path)
        errors.extend(
            _audit_manifest(
                manifest_rows,
                require_real_sha256=bool(cfg.get("require_real_sha256", False)),
            )
        )
    if pred_path.is_file():
        pred_rows = read_csv_dicts(pred_path)
        errors.extend(_audit_predictions(pred_rows))

    # Split consistency: prediction sample_ids must exist in manifest for same split
    if manifest_rows and pred_rows:
        man_index = {(r["sample_id"], r["split"]): r for r in manifest_rows if "sample_id" in r}
        for i, r in enumerate(pred_rows):
            key = (r.get("sample_id", ""), r.get("split", ""))
            if key not in man_index:
                errors.append(f"prediction row {i}: sample_id/split not in manifest: {key}")
            else:
                try:
                    if int(man_index[key]["binary_label"]) != int(r["y_true"]):
                        errors.append(f"prediction row {i}: y_true disagrees with manifest binary_label")
                except (KeyError, ValueError, TypeError):
                    pass

    audit: Dict[str, Any] = {
        "run_id": run_id,
        "claim_ids": cfg.get("claim_ids", []),
        "config_path": relpath_or_abs(config_path),
        "manifest_path": relpath_or_abs(manifest_path),
        "predictions_path": relpath_or_abs(pred_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "predictions_sha256": sha256_file(pred_path) if pred_path.is_file() else None,
        "task_level": task_level,
        "threshold_policy": threshold_policy,
        "git_dirty": dirty,
        "class_balance_manifest": _balance_report(manifest_rows, "binary_label"),
        "class_balance_predictions": _balance_report(pred_rows, "y_true"),
        "blockers": blockers,
        "errors": errors,
        "warnings": warnings,
        "passed": not errors and not blockers,
        "environment": environment_snapshot(),
    }

    write_json(out_dir / "input_audit.json", audit)
    md_lines = [
        f"# Input audit `{run_id}`",
        "",
        f"- passed: **{audit['passed']}**",
        f"- task_level: `{task_level}`",
        f"- threshold_policy: `{threshold_policy}`",
        f"- git_dirty: `{dirty}`",
        "",
        "## Blockers",
    ]
    md_lines.extend([f"- {b}" for b in blockers] or ["- (none)"])
    md_lines.append("")
    md_lines.append("## Errors")
    md_lines.extend([f"- {e}" for e in errors] or ["- (none)"])
    write_text(out_dir / "input_audit.md", "\n".join(md_lines) + "\n")
    copy_config_sidecar(config_path, out_dir)
    write_json(out_dir / "environment.json", audit["environment"])

    print(f"Wrote {out_dir / 'input_audit.json'}")
    if not audit["passed"]:
        print("AUDIT FAILED (fail closed)", file=sys.stderr)
        for b in blockers:
            print(f"  blocker: {b}", file=sys.stderr)
        for e in errors:
            print(f"  error: {e}", file=sys.stderr)
        return 2
    print("AUDIT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
