"""Generate ARTPS full-profile inference cache (train/validation only)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common import REPO_ROOT, read_csv_dicts, resolve_under_dataset_root, write_json  # noqa: E402
from artps_full_profile_cache import (  # noqa: E402
    allowed_splits,
    build_cache_index,
    cache_dir_for_profile,
    load_profile_yaml,
    profile_to_frozen_kwargs,
    verify_profile_registry,
    write_cache_entry,
    write_cache_metadata,
)
from src.artps_inference import FrozenARTPSConfig, load_frozen_artps_profile, predict_image  # noqa: E402
from test_split_embargo import assert_split_allowed  # noqa: E402


def _dataset_root(profile: dict) -> Path:
    env_key = str(profile.get("dataset_root_env", "ARTPS_DATASET_ROOT"))
    raw = os.environ.get(env_key)
    if not raw:
        raise RuntimeError(f"{env_key} is not set")
    return Path(raw).resolve()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate ARTPS full-profile cache")
    ap.add_argument("--profile", required=True, help="Profile YAML path")
    ap.add_argument("--force", action="store_true", help="Overwrite stale cache entries")
    ap.add_argument("--split", default=None, help="Optional single split (train|validation)")
    args = ap.parse_args(argv)

    profile = load_profile_yaml(args.profile)
    reg_errors = verify_profile_registry(profile)
    if reg_errors:
        for err in reg_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    splits = set(allowed_splits(profile))
    if args.split:
        assert_split_allowed(args.split)
        if args.split not in splits:
            print(f"ERROR: split {args.split!r} not in profile allowed_splits", file=sys.stderr)
            return 2
        splits = {args.split}

    cache_dir = cache_dir_for_profile(profile)
    write_cache_metadata(cache_dir, profile)

    manifest_path = REPO_ROOT / profile["dataset_manifest"]
    rows = read_csv_dicts(manifest_path)
    dataset_root = _dataset_root(profile)

    frozen_cfg = FrozenARTPSConfig(**profile_to_frozen_kwargs(profile))
    bundle = load_frozen_artps_profile(frozen_cfg)

    written = 0
    for row in rows:
        split = str(row.get("split", ""))
        if split not in splits:
            continue
        assert_split_allowed(split)
        sample_id = str(row["sample_id"])
        rel = str(row["relative_path"])
        image_path = resolve_under_dataset_root(dataset_root, rel)
        record = predict_image(
            image_path,
            bundle,
            frozen_cfg,
            sample_id=sample_id,
            split=split,
        )
        write_cache_entry(cache_dir, profile, sample_id, record, force=args.force)
        written += 1

    index = build_cache_index(cache_dir)
    write_json(cache_dir / "cache_index.json", index)
    print(f"OK: cached {written} samples under {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
