"""Frozen checkpoint registry verification for independent_eval_v1."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "reproduction" / "iac2026" / "frozen_checkpoint_registry.yaml"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(entry: Mapping[str, Any]) -> Path | None:
    raw = entry.get("path")
    if raw:
        p = Path(str(raw))
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()
    win_hint = entry.get("local_cache_path_windows")
    if win_hint:
        expanded = os.path.expandvars(str(win_hint))
        return Path(expanded).resolve()
    env_hint = entry.get("path_env_hint")
    if env_hint:
        parts = str(env_hint).replace("\\", "/").split("/")
        if parts[0] == "TORCH_HOME":
            base = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
            rel = Path(*parts[1:])
            return (base / rel).resolve()
    return None


def load_registry(path: Path | None = None) -> dict[str, Any]:
    reg_path = path or DEFAULT_REGISTRY
    with reg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"registry root must be mapping: {reg_path}")
    return data


def verify_registry_entry(entry: Mapping[str, Any], *, load_models: bool = False) -> list[str]:
    errors: list[str] = []
    cid = str(entry.get("checkpoint_id", "?"))
    path = _resolve_path(entry)
    if path is None:
        errors.append(f"{cid}: no resolvable path")
        return errors
    if not path.is_file():
        errors.append(f"{cid}: missing file {path}")
        return errors

    expected_size = entry.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        errors.append(
            f"{cid}: size mismatch expected={expected_size} actual={path.stat().st_size}"
        )

    expected_sha = str(entry.get("sha256") or "").strip().lower()
    if expected_sha:
        actual_sha = _sha256_file(path).lower()
        if actual_sha != expected_sha:
            errors.append(f"{cid}: sha256 mismatch expected={expected_sha} actual={actual_sha}")

    if not load_models:
        return errors

    model_type = str(entry.get("model_type", ""))
    arch = entry.get("expected_architecture") or {}
    state_key = str(arch.get("state_dict_key", "model_state_dict"))

    try:
        import torch  # local: CI software-verification does not install torch

        if model_type == "OptimizedAutoencoder":
            from src.models.optimized_autoencoder import OptimizedAutoencoder

            model = OptimizedAutoencoder(
                input_channels=int(arch.get("input_channels", 3)),
                latent_dim=int(arch.get("latent_dim", 1024)),
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(ckpt[state_key], strict=True)
            if missing or unexpected:
                errors.append(f"{cid}: AE strict load missing={missing} unexpected={unexpected}")
        elif model_type == "DepthEnhancedClassifier":
            from src.models.depth_enhanced_classifier import DepthEnhancedClassifier

            model = DepthEnhancedClassifier(
                num_classes=int(arch.get("num_classes", 5)),
                rgb_features=int(arch.get("rgb_features", 1024)),
                depth_features=int(arch.get("depth_features", 14)),
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(ckpt[state_key], strict=True)
            if missing or unexpected:
                errors.append(
                    f"{cid}: classifier strict load missing={missing} unexpected={unexpected}"
                )
        elif model_type == "DPT_Large":
            from src.models.depth_estimation import MiDaSDepthEstimator

            est = MiDaSDepthEstimator(model_type="DPT_Large", device="cpu", strict_local_only=True)
            if not est.is_real_dpt or est.load_source != "local_state_dict":
                errors.append(
                    f"{cid}: DPT strict load failed source={est.load_source!r} real={est.is_real_dpt}"
                )
    except Exception as exc:
        errors.append(f"{cid}: load_models failed: {exc}")

    return errors


def verify_registry(
    registry: Mapping[str, Any] | None = None,
    *,
    load_models: bool = False,
    primary_only: bool = False,
) -> list[str]:
    reg = dict(registry or load_registry())
    errors: list[str] = []
    for entry in reg.get("checkpoints") or []:
        if not isinstance(entry, dict):
            errors.append("invalid checkpoint entry (not a mapping)")
            continue
        if primary_only and entry.get("primary_or_exploratory") != "primary":
            continue
        errors.extend(verify_registry_entry(entry, load_models=load_models))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify frozen checkpoint registry")
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to frozen_checkpoint_registry.yaml",
    )
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Strict-load AE/classifier and strict_local_only DPT",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Verify only primary_or_exploratory=primary entries",
    )
    args = parser.parse_args(argv)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    reg = load_registry(args.registry)
    errors = verify_registry(reg, load_models=args.load_models, primary_only=args.primary_only)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        f"OK: verified {len(reg.get('checkpoints') or [])} checkpoint entries "
        f"(load_models={args.load_models}, primary_only={args.primary_only})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
