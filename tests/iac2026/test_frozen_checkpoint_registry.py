"""Frozen checkpoint registry verification (synthetic, no real weights)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from frozen_checkpoint_registry import load_registry, verify_registry, verify_registry_entry  # noqa: E402


def _write_fake_checkpoint(tmp_path: Path, name: str, payload: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def test_verify_registry_entry_sha_mismatch(tmp_path):
    ckpt = _write_fake_checkpoint(tmp_path, "ae.pth", b"not-the-real-checkpoint")
    entry = {
        "checkpoint_id": "ae_test",
        "path": str(ckpt),
        "size_bytes": ckpt.stat().st_size,
        "sha256": "0" * 64,
        "model_type": "OptimizedAutoencoder",
    }
    errors = verify_registry_entry(entry, load_models=False)
    assert any("sha256 mismatch" in e for e in errors)


def test_verify_registry_missing_file(tmp_path):
    entry = {
        "checkpoint_id": "missing",
        "path": str(tmp_path / "missing.pth"),
        "sha256": "a" * 64,
    }
    errors = verify_registry_entry(entry, load_models=False)
    assert any("missing file" in e for e in errors)


def test_primary_only_filters_exploratory(tmp_path):
    primary_ckpt = _write_fake_checkpoint(tmp_path, "primary.pth", b"primary-bytes")
    exploratory_ckpt = _write_fake_checkpoint(tmp_path, "explore.pth", b"explore-bytes")
    reg = {
        "checkpoints": [
            {
                "checkpoint_id": "primary",
                "path": str(primary_ckpt),
                "size_bytes": primary_ckpt.stat().st_size,
                "sha256": __import__("hashlib").sha256(b"primary-bytes").hexdigest(),
                "primary_or_exploratory": "primary",
            },
            {
                "checkpoint_id": "explore",
                "path": str(exploratory_ckpt),
                "size_bytes": exploratory_ckpt.stat().st_size,
                "sha256": "deadbeef" * 8,
                "primary_or_exploratory": "exploratory",
            },
        ]
    }
    errors = verify_registry(reg, load_models=False, primary_only=True)
    assert not any("explore" in e for e in errors)


def test_checked_in_registry_loads():
    reg = load_registry(REPO / "reproduction/iac2026/frozen_checkpoint_registry.yaml")
    assert reg["protocol_id"] == "independent_eval_v1"
    assert len(reg.get("checkpoints") or []) >= 5
