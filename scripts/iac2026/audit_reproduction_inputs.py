"""Fail-closed audit of IAC reproduction inputs (pure function + CLI)."""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    REPO_ROOT,
    environment_snapshot,
    git_dirty,
    git_head,
    load_json_schema,
    make_run_id,
    read_csv_dicts,
    relpath_or_abs,
    resolve_repo_path,
    resolve_under_dataset_root,
    sha256_file,
    validate_rows,
    write_json,
    write_run_bundle,
    write_text,
)
from _config import ConfigValidationError, LoadedConfig, load_and_validate_config


@dataclass
class AuditResult:
    passed: bool
    blockers: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    config_sha256: Optional[str] = None
    manifest_sha256: Optional[str] = None
    predictions_sha256: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    git_head: Optional[str] = None
    git_dirty: Optional[bool] = None
    evidence_mode: Optional[str] = None
    claim_ids: List[str] = field(default_factory=list)
    config_id: Optional[str] = None
    class_balance_manifest: Dict[str, Any] = field(default_factory=dict)
    class_balance_predictions: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compare_keys(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "evidence_mode": self.evidence_mode,
            "claim_ids": list(self.claim_ids),
            "config_id": self.config_id,
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "predictions_sha256": self.predictions_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "git_head": self.git_head,
            "git_dirty": self.git_dirty,
        }


def _balance(rows: List[Dict[str, str]], label_key: str) -> Dict[str, Any]:
    by_split: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        try:
            by_split[r.get("split", "")][int(r[label_key])] += 1
        except (KeyError, ValueError, TypeError):
            continue
    return {s: dict(c) for s, c in sorted(by_split.items())}


def _split_has_both_classes(rows: List[Dict[str, str]], split: str, label_key: str) -> bool:
    labels: Set[int] = set()
    for r in rows:
        if r.get("split") != split:
            continue
        try:
            labels.add(int(r[label_key]))
        except (KeyError, ValueError, TypeError):
            continue
    return 0 in labels and 1 in labels


def audit_inputs(
    cfg: LoadedConfig | Mapping[str, Any],
    *,
    config_sha256: Optional[str] = None,
    software_verification: bool = False,
) -> AuditResult:
    """Pure audit. Does not write files."""
    if isinstance(cfg, LoadedConfig):
        data = cfg.data
        config_sha = cfg.sha256
        config_path = cfg.path
    else:
        data = dict(cfg)
        config_sha = config_sha256
        config_path = None

    result = AuditResult(
        passed=False,
        config_sha256=config_sha,
        git_head=git_head(),
        git_dirty=git_dirty(),
        evidence_mode=str(data.get("evidence_mode")),
        claim_ids=list(data.get("claim_ids") or []),
        config_id=str(data.get("config_id") or ""),
    )
    real = data.get("evidence_mode") == "real_evidence"
    errors = result.errors
    blockers = result.blockers
    warnings = result.warnings

    if real and software_verification:
        blockers.append("real_evidence forbids --software-verification")

    if data.get("task_level") == "TASK_LEVEL_TBD" and real:
        blockers.append("task_level is TASK_LEVEL_TBD")

    dirty = bool(result.git_dirty)
    if dirty and not bool(data.get("allow_dirty_git", False)):
        errors.append("git working tree is dirty")

    manifest_path = resolve_repo_path(str(data["dataset_manifest"]))
    pred_path = resolve_repo_path(str(data["predictions_csv"]))
    if not manifest_path.is_file():
        errors.append(f"manifest missing: {manifest_path}")
    if not pred_path.is_file():
        errors.append(f"predictions missing: {pred_path}")

    manifest_rows: List[Dict[str, str]] = []
    pred_rows: List[Dict[str, str]] = []
    if manifest_path.is_file():
        manifest_rows = read_csv_dicts(manifest_path)
        result.manifest_sha256 = sha256_file(manifest_path)
        errors.extend(
            validate_rows(manifest_rows, load_json_schema("dataset_manifest.schema.json"), coerce_ints=["binary_label"])
        )
    if pred_path.is_file():
        pred_rows = read_csv_dicts(pred_path)
        result.predictions_sha256 = sha256_file(pred_path)
        errors.extend(
            validate_rows(
                pred_rows,
                load_json_schema("prediction_table.schema.json"),
                coerce_ints=["y_true"],
                coerce_floats=["anomaly_score"],
            )
        )

    # Duplicate sample_id
    ids = [r.get("sample_id", "") for r in manifest_rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample_id in manifest")

    # Leakage / empty groups
    scene_splits: Dict[str, Set[str]] = defaultdict(set)
    dup_splits: Dict[str, Set[str]] = defaultdict(set)
    sha_splits: Dict[str, Set[str]] = defaultdict(set)
    source_splits: Dict[str, Set[str]] = defaultdict(set)
    for r in manifest_rows:
        split = r.get("split", "")
        scene = r.get("scene_group_id", "")
        dup = r.get("duplicate_group_id", "")
        digest = r.get("sha256", "")
        source = r.get("source_id", "")
        if real and not scene:
            errors.append(f"empty scene_group_id for sample {r.get('sample_id')}")
        if real and not dup:
            errors.append(f"empty duplicate_group_id for sample {r.get('sample_id')}")
        if scene:
            scene_splits[scene].add(split)
        if dup:
            dup_splits[dup].add(split)
        if digest:
            sha_splits[digest].add(split)
        if source:
            source_splits[source].add(split)

    for scene, splits in scene_splits.items():
        if len(splits) > 1:
            errors.append(f"scene_group_id {scene!r} spans splits {sorted(splits)}")
    for dup, splits in dup_splits.items():
        if len(splits) > 1:
            errors.append(f"duplicate_group_id {dup!r} spans splits {sorted(splits)}")
    for digest, splits in sha_splits.items():
        if len(splits) > 1:
            errors.append(f"sha256 {digest[:12]}... spans splits {sorted(splits)}")
    cross = str(data.get("source_id_cross_split", "warning"))
    for source, splits in source_splits.items():
        if len(splits) > 1:
            msg = f"source_id {source!r} spans splits {sorted(splits)}"
            if cross == "error":
                errors.append(msg)
            else:
                warnings.append(msg)

    # Real file + SHA verification
    require_real = bool(data.get("require_real_sha256", False)) or real
    if require_real and manifest_rows:
        root_env = str(data.get("dataset_root_env", ""))
        root_val = os.environ.get(root_env)
        if not root_val:
            errors.append(f"dataset_root_env {root_env!r} is not set")
        else:
            dataset_root = Path(root_val)
            if not dataset_root.is_dir():
                errors.append(f"dataset root is not a directory: {dataset_root}")
            else:
                for i, r in enumerate(manifest_rows):
                    digest = r.get("sha256", "")
                    if digest.startswith("SYNTHETIC_") or len(digest) != 64:
                        errors.append(f"row {i}: sha256 must be 64 hex for real evidence")
                        continue
                    try:
                        path = resolve_under_dataset_root(dataset_root, r.get("relative_path", ""))
                    except ValueError as exc:
                        errors.append(f"row {i}: {exc}")
                        continue
                    if not path.is_file():
                        errors.append(f"row {i}: missing file {path}")
                        continue
                    actual = sha256_file(path)
                    if actual.lower() != digest.lower():
                        errors.append(f"row {i}: sha256 mismatch for {path.name}")

    # Checkpoint
    if real:
        ckpt = data.get("checkpoint_path")
        if not ckpt:
            errors.append("checkpoint_path required")
        else:
            ckpt_path = resolve_repo_path(str(ckpt))
            if not ckpt_path.is_file():
                errors.append(f"checkpoint missing: {ckpt_path}")
            else:
                actual = sha256_file(ckpt_path)
                result.checkpoint_sha256 = actual
                expected = str(data.get("checkpoint_sha256", ""))
                if actual.lower() != expected.lower():
                    errors.append("checkpoint_sha256 mismatch")

    # Predictions integrity
    if pred_rows:
        seen_keys: Set[Tuple[str, str, str, str, str]] = set()
        model_set: Set[Tuple[str, str, str]] = set()
        for i, r in enumerate(pred_rows):
            key = (
                r.get("sample_id", ""),
                r.get("split", ""),
                r.get("model_name", ""),
                r.get("model_version", ""),
                r.get("config_id", ""),
            )
            if key in seen_keys:
                errors.append(f"duplicate prediction key {key}")
            seen_keys.add(key)
            model_set.add((r.get("model_name", ""), r.get("model_version", ""), r.get("config_id", "")))
            try:
                score = float(r["anomaly_score"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"row {i}: anomaly_score not float")
                continue
            if not math.isfinite(score):
                errors.append(f"row {i}: anomaly_score not finite ({score})")
        if len(model_set) > 1:
            errors.append(f"mixed model/config in one prediction table: {sorted(model_set)}")

        expected_model = (
            str(data.get("model_name")),
            str(data.get("model_version")),
            str(data.get("config_id")),
        )
        for triple in model_set:
            if triple != expected_model:
                errors.append(
                    f"prediction model_name/model_version/config_id {triple} != config {expected_model}"
                )

    if manifest_rows and pred_rows:
        man_index = {(r["sample_id"], r["split"]): r for r in manifest_rows if "sample_id" in r}
        for i, r in enumerate(pred_rows):
            key = (r.get("sample_id", ""), r.get("split", ""))
            if key not in man_index:
                errors.append(f"prediction row {i}: not in manifest: {key}")
            else:
                try:
                    if int(man_index[key]["binary_label"]) != int(r["y_true"]):
                        errors.append(f"prediction row {i}: y_true disagrees with manifest")
                except (KeyError, ValueError, TypeError):
                    pass

        test_split = str(data.get("test_split", "test"))
        val_split = str(data.get("validation_split", "validation"))
        test_ids = {r["sample_id"] for r in manifest_rows if r.get("split") == test_split}
        pred_test = {r["sample_id"] for r in pred_rows if r.get("split") == test_split}
        missing_test = test_ids - pred_test
        if missing_test:
            errors.append(f"missing predictions for test samples: {sorted(missing_test)[:5]}")
        if data.get("threshold_policy") == "validation_selected":
            val_ids = {r["sample_id"] for r in manifest_rows if r.get("split") == val_split}
            pred_val = {r["sample_id"] for r in pred_rows if r.get("split") == val_split}
            missing_val = val_ids - pred_val
            if missing_val:
                errors.append(f"missing predictions for validation samples: {sorted(missing_val)[:5]}")
            if not _split_has_both_classes(pred_rows, val_split, "y_true"):
                errors.append("validation split must contain both classes for validation_selected")

        if not _split_has_both_classes(pred_rows, test_split, "y_true"):
            errors.append("test split must contain both positive and negative classes")

    result.class_balance_manifest = _balance(manifest_rows, "binary_label")
    result.class_balance_predictions = _balance(pred_rows, "y_true")
    result.details = {
        "config_path": relpath_or_abs(config_path) if config_path else None,
        "manifest_path": relpath_or_abs(manifest_path) if manifest_path.is_file() else None,
        "predictions_path": relpath_or_abs(pred_path) if pred_path.is_file() else None,
        "environment": environment_snapshot(),
    }
    result.passed = not errors and not blockers
    return result


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit IAC 2026 reproduction inputs (fail closed).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--software-verification", action="store_true")
    args = ap.parse_args(argv)

    try:
        loaded = load_and_validate_config(args.config)
    except ConfigValidationError as exc:
        print(f"CONFIG VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2

    if loaded.evidence_mode == "software_verification" and not args.software_verification:
        # Allow audit CLI without flag for SW configs, but record mode.
        pass

    result = audit_inputs(loaded, software_verification=args.software_verification)
    run_id = args.run_id or make_run_id("audit")
    out_dir = resolve_repo_path(str(loaded["output_directory"])) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["run_id"] = run_id
    write_json(out_dir / "input_audit.json", payload)
    write_text(
        out_dir / "input_audit.md",
        "\n".join(
            [
                f"# Input audit `{run_id}`",
                "",
                f"- passed: **{result.passed}**",
                f"- evidence_mode: `{result.evidence_mode}`",
                "",
                "## Blockers",
                *([f"- {b}" for b in result.blockers] if result.blockers else ["- (none)"]),
                "",
                "## Errors",
                *([f"- {e}" for e in result.errors] if result.errors else ["- (none)"]),
                "",
            ]
        ),
    )
    write_run_bundle(out_dir, config_path=loaded.path, command=sys.argv)
    print(f"Wrote {out_dir / 'input_audit.json'}")
    if not result.passed:
        print("AUDIT FAILED (fail closed)", file=sys.stderr)
        for b in result.blockers:
            print(f"  blocker: {b}", file=sys.stderr)
        for e in result.errors:
            print(f"  error: {e}", file=sys.stderr)
        return 2
    print("AUDIT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
