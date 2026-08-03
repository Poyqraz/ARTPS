"""Config load + schema validation for IAC reproduction (jsonschema required)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from _common import REPO_ROOT, load_json_schema, load_yaml, resolve_repo_path, sha256_file


class ConfigValidationError(ValueError):
    """Fail-closed config / schema errors (CLI should exit 2)."""


@dataclass
class LoadedConfig:
    path: Path
    data: Dict[str, Any]
    sha256: str

    @property
    def evidence_mode(self) -> str:
        return str(self.data["evidence_mode"])

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


def _is_tbd(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.upper().endswith("_TBD") or s.upper() == "TBD" or s.upper() == "UNKNOWN"


def require_jsonschema():
    try:
        import jsonschema
    except ImportError as exc:
        raise ConfigValidationError(
            "jsonschema is required for evidence harness validation; install "
            "reproduction/iac2026/requirements-ci.txt"
        ) from exc
    return jsonschema


def validate_instance(instance: Mapping[str, Any], schema_name: str) -> List[str]:
    jsonschema = require_jsonschema()
    schema = load_json_schema(schema_name)
    validator = jsonschema.Draft202012Validator(schema)
    return [e.message for e in sorted(validator.iter_errors(instance), key=lambda e: list(e.path))]


def apply_real_evidence_policy(cfg: Mapping[str, Any]) -> List[str]:
    """Extra real_evidence constraints beyond JSON Schema enums."""
    errors: List[str] = []
    if cfg.get("evidence_mode") != "real_evidence":
        return errors
    if cfg.get("task_level") == "TASK_LEVEL_TBD":
        errors.append("real_evidence forbids task_level=TASK_LEVEL_TBD")
    for key in ("model_name", "model_version", "checkpoint_sha256", "preprocessing", "normalization"):
        if _is_tbd(cfg.get(key)):
            errors.append(f"real_evidence forbids TBD/unknown {key}")
    if _is_tbd(cfg.get("resolution")) or str(cfg.get("resolution")).upper() == "N/A":
        errors.append("real_evidence forbids TBD/n/a resolution")
    if cfg.get("threshold_policy") == "unknown":
        errors.append("real_evidence forbids threshold_policy=unknown")
    if cfg.get("pr_metric_method") == "UNKNOWN":
        errors.append("real_evidence forbids pr_metric_method=UNKNOWN (C05 closure blocker)")
    if cfg.get("require_real_sha256") is not True:
        errors.append("real_evidence requires require_real_sha256=true")
    if cfg.get("allow_dirty_git") is not False:
        errors.append("real_evidence requires allow_dirty_git=false")
    if not cfg.get("checkpoint_path"):
        errors.append("real_evidence requires checkpoint_path")
    return errors


def load_and_validate_config(path: Path | str) -> LoadedConfig:
    config_path = resolve_repo_path(str(path)) if not isinstance(path, Path) else path
    if not config_path.is_file():
        raise ConfigValidationError(f"config not found: {config_path}")
    try:
        data = load_yaml(config_path)
    except Exception as exc:
        raise ConfigValidationError(f"failed to parse YAML: {exc}") from exc

    schema_errors = validate_instance(data, "detection_reproduction_config.schema.json")
    if schema_errors:
        raise ConfigValidationError("config schema validation failed:\n- " + "\n- ".join(schema_errors))

    policy_errors = apply_real_evidence_policy(data)
    if policy_errors:
        raise ConfigValidationError("real_evidence policy failed:\n- " + "\n- ".join(policy_errors))

    if int(data.get("bootstrap_iterations", 0)) != 0:
        raise ConfigValidationError(
            "bootstrap_iterations != 0 is not implemented; set to 0 (do not ignore silently)"
        )

    return LoadedConfig(path=config_path, data=data, sha256=sha256_file(config_path))


def load_timing_config(path: Path | str) -> Dict[str, Any]:
    """Timing configs are lighter; validate required keys in code."""
    config_path = resolve_repo_path(str(path)) if not isinstance(path, Path) else path
    data = load_yaml(config_path)
    required = [
        "claim_ids",
        "evidence_mode",
        "profile",
        "input_resolution",
        "batch_size",
        "learned_depth_enabled",
        "autoencoder_enabled",
        "warmup_count",
        "timed_iteration_count",
        "output_directory",
        "allow_dirty_git",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ConfigValidationError(f"timing config missing keys: {missing}")
    return data
