"""Executable independent_eval_v1 protocol lock checks (fail-closed)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from _common import REPO_ROOT, load_yaml, resolve_repo_path

DEFAULT_LOCK_REL = "reproduction/iac2026/INDEPENDENT_EVAL_V1.yaml"

# Config key -> lock path (flat value or nested {"value": ...})
_LOCK_FIELD_MAP = (
    ("protocol_id", "protocol_id"),
    ("evaluation_purpose", "evaluation_purpose"),
    ("task_level", "task_level"),
    ("positive_label", "positive_label"),
    ("label_semantics", "label_semantics"),
    ("annotation_version", "annotation_version"),
    ("higher_score_means_more_anomalous", "higher_score_means_more_anomalous"),
    ("image_score_aggregation", "image_score_aggregation"),
    ("train_split", "train_split"),
    ("validation_split", "validation_split"),
    ("test_split", "test_split"),
    ("pr_metric_method", "pr_metric_method"),
    ("threshold_policy", "threshold_policy"),
    ("threshold_selection_metric", "threshold_selection_metric"),
    ("threshold_tie_break", "threshold_tie_break"),
    ("bootstrap_iterations", "bootstrap_iterations"),
)

REQUIRED_CONFIG_KEYS = (
    "protocol_id",
    "protocol_lock_path",
    "protocol_lock_sha256",
    "annotation_version",
    "image_score_aggregation",
    "label_semantics",
)


def sha256_protocol_lock_bytes(path: Path) -> str:
    """SHA256 of lock file with newlines normalized to LF (cross-platform)."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _lock_value(node: Any) -> Any:
    if isinstance(node, Mapping) and "value" in node:
        return node["value"]
    return node


def load_protocol_lock(path: Path | str | None = None) -> Tuple[Dict[str, Any], str, Path]:
    """Return (lock_dict, sha256_hex, resolved_path)."""
    if path is None:
        lock_path = REPO_ROOT / DEFAULT_LOCK_REL
    else:
        lock_path = resolve_repo_path(str(path)) if not isinstance(path, Path) else path
        if not lock_path.is_absolute():
            lock_path = resolve_repo_path(str(lock_path))
    if not lock_path.is_file():
        raise FileNotFoundError(f"protocol lock missing: {lock_path}")
    data = load_yaml(lock_path)
    if not isinstance(data, dict):
        raise ValueError(f"protocol lock must be a mapping: {lock_path}")
    return data, sha256_protocol_lock_bytes(lock_path), lock_path


def flatten_locked_values(lock: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "protocol_id": lock.get("protocol_id"),
        "evaluation_purpose": lock.get("evaluation_purpose"),
        "claim_ids": list(lock.get("claim_ids") or []),
    }
    for cfg_key, lock_key in _LOCK_FIELD_MAP:
        if lock_key in ("protocol_id", "evaluation_purpose"):
            continue
        if lock_key in lock:
            out[cfg_key] = _lock_value(lock[lock_key])
    # Alias positive_class -> positive_label if needed
    if "positive_label" not in out or out["positive_label"] is None:
        out["positive_label"] = _lock_value(lock.get("positive_class"))
    return out


def validate_config_against_lock(
    cfg: Mapping[str, Any],
    *,
    repo_root: Optional[Path] = None,
) -> List[str]:
    """Return error strings; empty means OK. Only for current_reproducible_evaluation."""
    errors: List[str] = []
    if cfg.get("evaluation_purpose") != "current_reproducible_evaluation":
        return errors

    for key in REQUIRED_CONFIG_KEYS:
        if key not in cfg or cfg.get(key) in (None, ""):
            errors.append(f"independent eval config missing required key: {key}")

    lock_rel = cfg.get("protocol_lock_path") or DEFAULT_LOCK_REL
    try:
        if repo_root is not None:
            lock_path = Path(lock_rel)
            if not lock_path.is_absolute():
                lock_path = (repo_root / lock_rel).resolve()
            lock, actual_sha, _ = load_protocol_lock(lock_path)
        else:
            lock, actual_sha, _ = load_protocol_lock(str(lock_rel))
    except (OSError, ValueError, FileNotFoundError) as exc:
        return errors + [f"protocol lock unreadable: {exc}"]

    declared = str(cfg.get("protocol_lock_sha256") or "").strip().lower()
    if not declared or len(declared) != 64:
        errors.append("protocol_lock_sha256 must be a 64-hex digest matching the lock file")
    elif declared != actual_sha.lower():
        errors.append(
            f"protocol_lock_sha256 mismatch: declared={declared} actual={actual_sha.lower()}"
        )

    expected = flatten_locked_values(lock)
    if cfg.get("protocol_id") != expected["protocol_id"]:
        errors.append(
            f"protocol_id must be {expected['protocol_id']!r} (got {cfg.get('protocol_id')!r})"
        )
    if cfg.get("evaluation_purpose") != expected["evaluation_purpose"]:
        errors.append("evaluation_purpose must match protocol lock")
    if list(cfg.get("claim_ids") or []) != list(expected["claim_ids"]):
        errors.append(f"claim_ids must equal {expected['claim_ids']} (got {cfg.get('claim_ids')})")

    for cfg_key, _lock_key in _LOCK_FIELD_MAP:
        if cfg_key in ("protocol_id", "evaluation_purpose"):
            continue
        exp = expected.get(cfg_key)
        got = cfg.get(cfg_key)
        if exp != got:
            errors.append(f"config {cfg_key}={got!r} does not match lock value {exp!r}")

    return errors


def protocol_provenance_from_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Fields to stamp onto audit/metrics for any config."""
    purpose = cfg.get("evaluation_purpose")
    if purpose == "current_reproducible_evaluation":
        return {
            "protocol_id": cfg.get("protocol_id"),
            "protocol_lock_sha256": cfg.get("protocol_lock_sha256"),
            "evaluation_purpose": purpose,
            "annotation_version": cfg.get("annotation_version"),
            "image_score_aggregation": cfg.get("image_score_aggregation"),
            "label_semantics": cfg.get("label_semantics"),
        }
    return {
        "protocol_id": None,
        "protocol_lock_sha256": None,
        "evaluation_purpose": purpose,
        "annotation_version": None,
        "image_score_aggregation": None,
        "label_semantics": cfg.get("label_semantics"),
    }
