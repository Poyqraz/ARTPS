"""Full-profile cache library tests (synthetic)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from artps_full_profile_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    build_cache_metadata,
    cache_dir_for_profile,
    read_cache_entry,
    verify_cache_metadata,
    write_cache_entry,
    write_cache_metadata,
)


@pytest.fixture
def profile(tmp_path):
    raw = yaml.safe_load(
        (REPO / "reproduction/iac2026/configs/independent_eval_artps_full_frozen.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["dataset_manifest"] = str(tmp_path / "manifest.csv")
    return raw


def test_cache_metadata_roundtrip(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "artps_full_profile_cache.REPO_ROOT",
        tmp_path,
        raising=False,
    )
    cache_dir = cache_dir_for_profile(profile, repo_root=tmp_path)
    write_cache_metadata(cache_dir, profile)
    assert verify_cache_metadata(cache_dir, profile) == []
    meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == CACHE_SCHEMA_VERSION
    assert meta["config_id"] == profile["config_id"]


def test_write_and_read_cache_entry(profile, tmp_path, monkeypatch):
    monkeypatch.setattr("artps_full_profile_cache.REPO_ROOT", tmp_path, raising=False)
    cache_dir = cache_dir_for_profile(profile, repo_root=tmp_path)
    write_cache_metadata(cache_dir, profile)
    record = {
        "image_score": 0.42,
        "top_candidate_score": 0.42,
        "candidate_count": 1,
        "split": "validation",
        "candidates": [{"x": 1, "y": 2, "w": 3, "h": 4, "score": 0.42}],
        "processing_status": "ok",
        "warning_flags": [],
    }
    write_cache_entry(cache_dir, profile, "sample_a", record)
    loaded = read_cache_entry(cache_dir, "sample_a")
    assert loaded is not None
    assert loaded["image_score"] == pytest.approx(0.42)
    assert loaded["candidates"][0]["score"] == pytest.approx(0.42)


def test_stale_cache_write_fail_closed(profile, tmp_path, monkeypatch):
    monkeypatch.setattr("artps_full_profile_cache.REPO_ROOT", tmp_path, raising=False)
    cache_dir = cache_dir_for_profile(profile, repo_root=tmp_path)
    write_cache_metadata(cache_dir, profile)
    stale = dict(profile)
    stale["precision"] = "amp"
    with pytest.raises(RuntimeError, match="stale"):
        write_cache_entry(
            cache_dir,
            stale,
            "sample_a",
            {"image_score": 0.1, "top_candidate_score": 0.1, "candidate_count": 0},
        )


def test_build_cache_metadata_includes_hashes(profile):
    meta = build_cache_metadata(profile)
    assert "8186" in meta["checkpoint_hashes"]["ae"]
