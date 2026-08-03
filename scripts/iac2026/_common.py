"""Shared I/O helpers for IAC 2026 reproduction harness."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def relpath_or_abs(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def git_dirty(cwd: Path = REPO_ROOT) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return True
    return bool(out.strip())


def git_head(cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "UNKNOWN"


def environment_snapshot() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "git_head": git_head(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def make_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def load_json_schema(name: str) -> Dict[str, Any]:
    schema_path = REPO_ROOT / "reproduction" / "iac2026" / "schemas" / name
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_jsonschema():
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "jsonschema is required; install reproduction/iac2026/requirements-ci.txt"
        ) from exc
    return jsonschema


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    *,
    coerce_ints: Optional[Iterable[str]] = None,
    coerce_floats: Optional[Iterable[str]] = None,
) -> List[str]:
    """Validate CSV-derived rows. Fail closed if jsonschema missing."""
    jsonschema = require_jsonschema()
    errors: List[str] = []
    coerced: List[Dict[str, Any]] = []
    int_keys = set(coerce_ints or ())
    float_keys = set(coerce_floats or ())
    for i, row in enumerate(rows):
        item: Dict[str, Any] = dict(row)
        for k in int_keys:
            if k in item and item[k] not in (None, ""):
                try:
                    item[k] = int(item[k])
                except (TypeError, ValueError):
                    errors.append(f"row {i}: {k} not int-compatible: {item[k]!r}")
        for k in float_keys:
            if k in item and item[k] not in (None, ""):
                try:
                    item[k] = float(item[k])
                except (TypeError, ValueError):
                    errors.append(f"row {i}: {k} not float-compatible: {item[k]!r}")
        coerced.append(item)

    validator = jsonschema.Draft202012Validator(schema)
    for i, item in enumerate(coerced):
        for err in sorted(validator.iter_errors(item), key=lambda e: e.path):
            errors.append(f"row {i}: {err.message}")
    return errors


def copy_config_sidecar(config_path: Path, dest_dir: Path) -> Path:
    dest = dest_dir / "config_used.yaml"
    dest.write_bytes(config_path.read_bytes())
    return dest


def write_run_bundle(
    out_dir: Path,
    *,
    config_path: Optional[Path],
    command: Sequence[str],
    provenance_extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment_snapshot()
    write_json(out_dir / "environment.json", env)
    if config_path is not None and config_path.is_file():
        copy_config_sidecar(config_path, out_dir)
        config_sha = sha256_file(config_path)
    else:
        config_sha = None
    provenance = {
        "git_head": env["git_head"],
        "git_dirty": git_dirty(),
        "config_sha256": config_sha,
        "command": " ".join(command),
        "timestamp_utc": env["timestamp_utc"],
    }
    if provenance_extra:
        provenance.update(dict(provenance_extra))
    write_json(out_dir / "provenance.json", provenance)
    write_text(out_dir / "command.txt", " ".join(command) + "\n")
    return provenance


def resolve_under_dataset_root(dataset_root: Path, relative_path: str) -> Path:
    """Reject absolute paths, .. traversal, and symlink escape outside root."""
    if not relative_path or relative_path.strip() == "":
        raise ValueError("empty relative_path")
    rel = Path(relative_path)
    if rel.is_absolute():
        raise ValueError(f"absolute relative_path rejected: {relative_path}")
    if ".." in rel.parts:
        raise ValueError(f"path traversal rejected: {relative_path}")
    root = dataset_root.resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes dataset root: {relative_path}") from exc
    return candidate
