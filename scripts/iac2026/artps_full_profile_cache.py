"""ARTPS full-profile inference cache (atomic writes, fail-closed on stale)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np
import yaml

from _common import REPO_ROOT, load_yaml, sha256_bytes, sha256_file, write_json
from frozen_checkpoint_registry import verify_registry
from test_split_embargo import assert_split_allowed

CACHE_SCHEMA_VERSION = 1
CACHE_ROOT_REL = "results/iac2026/independent_eval_v1/cache"

PROTOCOL_LOCK_SHA256 = (
    "7767f695746d0237803f57ffd2fef8f96a1434fca5d2f2ffaf2c799c3187dfe9"
)


def load_profile_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    data = load_yaml(p)
    _validate_profile_required(data)
    return data


def _validate_profile_required(profile: Mapping[str, Any]) -> None:
    required = (
        "config_id",
        "protocol_id",
        "dataset_manifest",
        "dataset_root_env",
        "preprocessing_profile",
        "autoencoder",
        "depth",
        "depth_classifier",
    )
    missing = [k for k in required if k not in profile]
    if missing:
        raise ValueError(f"profile missing required keys: {missing}")


def profile_config_sha256(profile: Mapping[str, Any]) -> str:
    payload = yaml.safe_dump(dict(profile), sort_keys=True, allow_unicode=True)
    return sha256_bytes(payload.encode("utf-8"))


def cache_dir_for_profile(profile: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> Path:
    config_id = str(profile["config_id"])
    config_sha = profile_config_sha256(profile)
    return repo_root / CACHE_ROOT_REL / config_id / config_sha


def profile_to_checkpoint_hashes(profile: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    ae = profile.get("autoencoder") or {}
    depth = profile.get("depth") or {}
    clf = profile.get("depth_classifier") or {}
    if ae.get("checkpoint_sha256"):
        out["ae"] = str(ae["checkpoint_sha256"]).lower()
    if depth.get("checkpoint_sha256"):
        out["dpt"] = str(depth["checkpoint_sha256"]).lower()
    if clf.get("enabled") and clf.get("checkpoint_sha256"):
        out["classifier"] = str(clf["checkpoint_sha256"]).lower()
    return out


def profile_to_frozen_kwargs(profile: Mapping[str, Any]) -> dict[str, Any]:
    ae = profile.get("autoencoder") or {}
    clf = profile.get("depth_classifier") or {}
    precision = str(profile.get("precision", "fp32")).lower()
    return {
        "config_id": str(profile["config_id"]),
        "protocol_id": str(profile.get("protocol_id", "independent_eval_v1")),
        "protocol_lock_path": str(
            profile.get("protocol_lock_path", "reproduction/iac2026/INDEPENDENT_EVAL_V1.yaml")
        ),
        "protocol_lock_sha256": str(
            profile.get("protocol_lock_sha256") or PROTOCOL_LOCK_SHA256
        ),
        "ae_path": str(ae.get("checkpoint", "results/optimized_autoencoder_curiosity_extended.pth")),
        "classifier_path": str(
            clf.get("checkpoint", "results/depth_enhanced_classifier.pth")
        ),
        "enable_classifier": bool(clf.get("enabled", True)),
        "preprocessing_profile": str(profile["preprocessing_profile"]),
        "use_amp": precision == "amp",
        "checkpoint_sha256": profile_to_checkpoint_hashes(profile),
    }


def build_metrics_config_snapshot(
    profile: Mapping[str, Any],
    *,
    predictions_csv: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Thin detection_reproduction_config-compatible dict for metrics validation."""
    return {
        "claim_ids": list(profile.get("claim_ids") or ["IND_EVAL_V1"]),
        "evidence_mode": "real_evidence",
        "evaluation_purpose": profile.get(
            "evaluation_purpose", "current_reproducible_evaluation"
        ),
        "protocol_id": profile.get("protocol_id", "independent_eval_v1"),
        "protocol_lock_path": profile.get(
            "protocol_lock_path", "reproduction/iac2026/INDEPENDENT_EVAL_V1.yaml"
        ),
        "protocol_lock_sha256": profile.get("protocol_lock_sha256") or PROTOCOL_LOCK_SHA256,
        "annotation_version": profile.get("annotation_version", "independent_eval_v1"),
        "label_semantics": profile.get("label_semantics", "anomaly_binary"),
        "image_score_aggregation": profile.get(
            "image_score_aggregation", "max_valid_candidate_after_masks"
        ),
        "task_level": profile.get("task_level", "image_binary"),
        "positive_label": int(profile.get("positive_label", 1)),
        "score_semantics": profile.get("score_semantics", "higher_score_means_more_anomalous"),
        "higher_score_means_more_anomalous": bool(
            profile.get("higher_score_means_more_anomalous", True)
        ),
        "dataset_manifest": str(profile["dataset_manifest"]),
        "dataset_root_env": str(profile.get("dataset_root_env", "ARTPS_DATASET_ROOT")),
        "train_split": profile.get("train_split", "train"),
        "validation_split": profile.get("validation_split", "validation"),
        "test_split": profile.get("test_split", "test"),
        "random_seed": int(profile.get("random_seed", 0)),
        "preprocessing": profile.get("preprocessing_profile", "raw_rgb_v1"),
        "resolution": int(profile.get("resolution", 256)),
        "normalization": profile.get("normalization", "imagenet"),
        "model_name": profile.get("model_name", "ARTPS"),
        "model_version": profile.get("model_version", "frozen_legacy_full_profile_v1"),
        "checkpoint_path": (profile.get("autoencoder") or {}).get("checkpoint"),
        "checkpoint_sha256": (profile.get("autoencoder") or {}).get("checkpoint_sha256"),
        "threshold_policy": profile.get("threshold_policy", "validation_selected"),
        "threshold_selection_metric": profile.get("threshold_selection_metric", "f1"),
        "threshold_tie_break": profile.get("threshold_tie_break", "highest_threshold"),
        "fixed_threshold": profile.get("fixed_threshold"),
        "pr_metric_method": profile.get("pr_metric_method", "average_precision"),
        "bootstrap_iterations": int(profile.get("bootstrap_iterations", 0)),
        "predictions_csv": str(predictions_csv),
        "output_directory": str(output_directory),
        "allow_dirty_git": bool(profile.get("allow_dirty_git", False)),
        "require_real_sha256": bool(profile.get("require_real_sha256", True)),
        "config_id": str(profile["config_id"]),
        "source_id_cross_split": profile.get("source_id_cross_split", "error"),
    }


def verify_profile_registry(profile: Mapping[str, Any]) -> list[str]:
    registry_path = profile.get("registry_path", "reproduction/iac2026/frozen_checkpoint_registry.yaml")
    reg_path = REPO_ROOT / registry_path if not Path(str(registry_path)).is_absolute() else Path(registry_path)
    reg = load_yaml(reg_path)
    errors = verify_registry(reg, load_models=False, primary_only=True)
    expected = profile_to_checkpoint_hashes(profile)
    by_id = {str(e.get("checkpoint_id")): e for e in reg.get("checkpoints") or [] if isinstance(e, dict)}
    key_map = {
        "ae": "ae_curiosity_extended",
        "dpt": "dpt_large_384",
        "classifier": "depth_enhanced_classifier",
    }
    for key, sha in expected.items():
        entry = by_id.get(key_map.get(key, ""))
        if entry is None:
            errors.append(f"registry missing entry for profile hash key {key}")
            continue
        reg_sha = str(entry.get("sha256") or "").lower()
        if reg_sha != sha.lower():
            errors.append(f"profile/registry sha mismatch for {key}: profile={sha} registry={reg_sha}")
    return errors


def allowed_splits(profile: Mapping[str, Any]) -> list[str]:
    splits = list(profile.get("allowed_splits") or ["train", "validation"])
    out: list[str] = []
    for s in splits:
        assert_split_allowed(str(s))
        out.append(str(s))
    return out


def _metadata_path(cache_dir: Path) -> Path:
    return cache_dir / "metadata.json"


def _sample_npz_path(cache_dir: Path, sample_id: str) -> Path:
    safe = sample_id.replace("/", "_")
    return cache_dir / "samples" / f"{safe}.npz"


def _sample_candidates_path(cache_dir: Path, sample_id: str) -> Path:
    safe = sample_id.replace("/", "_")
    return cache_dir / "samples" / f"{safe}.candidates.json"


def build_cache_metadata(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "config_id": str(profile["config_id"]),
        "config_sha256": profile_config_sha256(profile),
        "protocol_id": profile.get("protocol_id"),
        "protocol_lock_sha256": profile.get("protocol_lock_sha256") or PROTOCOL_LOCK_SHA256,
        "preprocessing_profile": profile.get("preprocessing_profile"),
        "precision": profile.get("precision", "fp32"),
        "checkpoint_hashes": profile_to_checkpoint_hashes(profile),
        "image_score_aggregation": profile.get("image_score_aggregation"),
    }


def verify_cache_metadata(cache_dir: Path, profile: Mapping[str, Any]) -> list[str]:
    meta_path = _metadata_path(cache_dir)
    if not meta_path.is_file():
        return ["cache metadata missing"]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"cache metadata invalid json: {exc}"]
    errors: list[str] = []
    expected = build_cache_metadata(profile)
    for key in (
        "schema_version",
        "config_id",
        "config_sha256",
        "protocol_lock_sha256",
        "preprocessing_profile",
        "precision",
        "checkpoint_hashes",
    ):
        if meta.get(key) != expected.get(key):
            errors.append(f"cache stale: metadata {key} mismatch")
    return errors


def write_cache_metadata(cache_dir: Path, profile: Mapping[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = build_cache_metadata(profile)
    tmp = _metadata_path(cache_dir).with_suffix(".json.tmp")
    write_json(tmp, meta)
    tmp.replace(_metadata_path(cache_dir))


def write_cache_entry(
    cache_dir: Path,
    profile: Mapping[str, Any],
    sample_id: str,
    record: Mapping[str, Any],
    *,
    force: bool = False,
) -> None:
    stale = verify_cache_metadata(cache_dir, profile) if _metadata_path(cache_dir).is_file() else []
    if stale and not force:
        raise RuntimeError("refusing stale cache write: " + "; ".join(stale))

    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    npz_path = _sample_npz_path(cache_dir, sample_id)
    cand_path = _sample_candidates_path(cache_dir, sample_id)
    tmp_npz = npz_path.with_name(f"{npz_path.stem}.tmp.npz")
    tmp_cand = cand_path.with_suffix(".json.tmp")

    image_score = float(record.get("image_score", 0.0))
    top_score = float(record.get("top_candidate_score", image_score))
    np.savez_compressed(
        str(tmp_npz),
        image_score=np.float64(image_score),
        top_candidate_score=np.float64(top_score),
        candidate_count=np.int32(int(record.get("candidate_count", 0))),
    )
    cand_payload: MutableMapping[str, Any] = {
        "sample_id": sample_id,
        "split": record.get("split"),
        "candidates": record.get("candidates") or [],
        "processing_status": record.get("processing_status", "ok"),
        "warning_flags": list(record.get("warning_flags") or []),
    }
    write_json(tmp_cand, cand_payload)
    tmp_npz.replace(npz_path)
    tmp_cand.replace(cand_path)


def read_cache_entry(cache_dir: Path, sample_id: str) -> dict[str, Any] | None:
    npz_path = _sample_npz_path(cache_dir, sample_id)
    cand_path = _sample_candidates_path(cache_dir, sample_id)
    if not npz_path.is_file():
        return None
    data = np.load(npz_path)
    out: dict[str, Any] = {
        "sample_id": sample_id,
        "image_score": float(data["image_score"]),
        "top_candidate_score": float(data["top_candidate_score"]),
        "candidate_count": int(data["candidate_count"]),
    }
    if cand_path.is_file():
        out.update(json.loads(cand_path.read_text(encoding="utf-8")))
    return out


def build_cache_index(cache_dir: Path) -> dict[str, Any]:
    samples_dir = cache_dir / "samples"
    entries: list[dict[str, Any]] = []
    if samples_dir.is_dir():
        for npz in sorted(samples_dir.glob("*.npz")):
            sample_key = npz.stem
            entries.append(
                {
                    "sample_id_key": sample_key,
                    "npz_sha256": sha256_file(npz),
                    "npz_path": str(npz.relative_to(cache_dir)).replace("\\", "/"),
                }
            )
    meta_sha = sha256_file(_metadata_path(cache_dir)) if _metadata_path(cache_dir).is_file() else None
    return {
        "cache_dir": str(cache_dir),
        "metadata_sha256": meta_sha,
        "sample_count": len(entries),
        "entries": entries,
    }


def environment_snapshot_torch() -> dict[str, Any]:
    from _common import environment_snapshot

    env = environment_snapshot()
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device_count"] = torch.cuda.device_count()
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch_version"] = None
    env["dataset_root"] = os.environ.get("ARTPS_DATASET_ROOT")
    return env
