"""Shared I/O helpers for IAC 2026 reproduction harness (stdlib-first)."""
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
        json.dump(payload, f, indent=2, sort_keys=True)
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


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    *,
    coerce_ints: Optional[Iterable[str]] = None,
    coerce_floats: Optional[Iterable[str]] = None,
) -> List[str]:
    """Validate CSV-derived rows against a JSON Schema object definition.

    Uses jsonschema when available; otherwise applies required-key checks only.
    """
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

    try:
        import jsonschema
    except ImportError:
        required = list(schema.get("required") or [])
        for i, item in enumerate(coerced):
            for key in required:
                if key not in item or item[key] in (None, ""):
                    errors.append(f"row {i}: missing required field {key}")
        return errors

    validator = jsonschema.Draft202012Validator(schema)
    for i, item in enumerate(coerced):
        for err in sorted(validator.iter_errors(item), key=lambda e: e.path):
            errors.append(f"row {i}: {err.message}")
    return errors


def copy_config_sidecar(config_path: Path, dest_dir: Path) -> Path:
    dest = dest_dir / "config_used.yaml"
    dest.write_bytes(config_path.read_bytes())
    return dest
