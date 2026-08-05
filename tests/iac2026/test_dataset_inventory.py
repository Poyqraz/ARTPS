"""Tests for independent_eval dataset inventory fail-closed behaviour."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from inventory_independent_eval_dataset import (  # noqa: E402
    DATASET_ROOT_MSG,
    build_inventory,
    require_dataset_root,
    sha256_file,
)


def test_missing_dataset_root_fail_closed(monkeypatch):
    monkeypatch.delenv("ARTPS_DATASET_ROOT", raising=False)
    with pytest.raises(SystemExit) as ei:
        require_dataset_root()
    assert ei.value.code == 2


def test_missing_dataset_root_message(monkeypatch, capsys):
    monkeypatch.delenv("ARTPS_DATASET_ROOT", raising=False)
    with pytest.raises(SystemExit):
        require_dataset_root()
    err = capsys.readouterr().err
    assert "DATASET ROOT REQUIRED" in err


def test_unreadable_file_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTPS_DATASET_ROOT", str(tmp_path))
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not-an-image")
    rows = build_inventory(tmp_path)
    assert len(rows) == 1
    assert rows[0]["readable"] == "false"
    assert "unreadable" in rows[0]["quality_flags"]


def test_duplicate_sha_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTPS_DATASET_ROOT", str(tmp_path))
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(10, 20, 30))
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    img.save(a)
    img.save(b)
    assert sha256_file(a) == sha256_file(b)
    rows = build_inventory(tmp_path)
    assert len(rows) == 2
    assert rows[0]["duplicate_candidate_group"] == rows[1]["duplicate_candidate_group"]
    assert all("exact_sha_duplicate" in r["quality_flags"] for r in rows)
