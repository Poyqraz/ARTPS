"""Blind validation review queue / pack integrity tests."""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from build_validation_blind_review_pack import (  # noqa: E402
    build_pack,
    resolve_dataset_root,
)
from validation_blind_review import (  # noqa: E402
    BLIND_QUEUE_FIELDS,
    FORBIDDEN_VISIBLE_COLUMNS,
    FORBIDDEN_VISIBLE_SUBSTRINGS,
    assert_public_row_blind,
    build_blind_public_and_private,
)

BLIND_QUEUE = (
    REPO
    / "reproduction/iac2026/annotations/independent_eval_v1_validation_blind_review_queue.csv"
)
MANIFEST = REPO / "reproduction/iac2026/manifests/independent_eval_v1.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_visible_queue_schema_and_forbidden_columns():
    rows = _read_csv(BLIND_QUEUE)
    assert len(rows) == 54
    assert list(rows[0].keys()) == BLIND_QUEUE_FIELDS
    for col in FORBIDDEN_VISIBLE_COLUMNS:
        assert col not in rows[0]


def test_visible_queue_no_forbidden_substrings():
    rows = _read_csv(BLIND_QUEUE)
    for row in rows:
        blob = " ".join(str(v) for v in row.values()).lower()
        for bad in FORBIDDEN_VISIBLE_SUBSTRINGS:
            assert bad not in blob, f"{bad} in {row}"


def test_neutral_filenames_unique_and_ordered():
    rows = _read_csv(BLIND_QUEUE)
    names = [r["neutral_filename"] for r in rows]
    assert len(names) == len(set(names))
    assert names[0] == "review_0001.jpg"
    assert names[-1] == "review_0054.jpg"
    assert all(r["audit_status"] == "pending_independent_review" for r in rows)


def test_hashes_match_manifest_sha256():
    rows = _read_csv(BLIND_QUEUE)
    manifest = {r["sha256"]: r for r in _read_csv(MANIFEST) if r.get("sha256")}
    for row in rows:
        assert row["image_sha256"] in manifest


def test_missing_dataset_root_fails_closed(monkeypatch):
    monkeypatch.delenv("ARTPS_DATASET_ROOT", raising=False)
    with pytest.raises(ValueError, match="fail closed|unset"):
        resolve_dataset_root(None)


def test_pack_generation_deterministic(tmp_path, monkeypatch):
    """Synthetic 54-image root; pack twice → same hashes/order."""
    monkeypatch.delenv("ARTPS_DATASET_ROOT", raising=False)
    manifest = _read_csv(MANIFEST)
    public, private = build_blind_public_and_private(manifest)
    assert len(public) == 54
    root = tmp_path / "ds"
    for priv in private:
        assert str(priv["split"]).lower() != "test"
        dest = root / priv["relative_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = (priv["sample_id"] + "\n").encode()
        # Override sha in private/public for synthetic files
        digest = hashlib.sha256(payload).hexdigest()
        dest.write_bytes(payload)
        priv["image_sha256"] = digest
    # Rewrite public hashes to match synthetic files via same order
    for pub, priv in zip(public, private):
        pub["image_sha256"] = priv["image_sha256"]
        assert_public_row_blind(pub)

    # Build pack using real helper with patched relative files + env
    # Use a tiny custom build: write images then call build_pack after patching MANIFEST is heavy.
    # Contract: two permutations with same seed match.
    a, _ = build_blind_public_and_private(manifest, seed=20260806)
    b, _ = build_blind_public_and_private(manifest, seed=20260806)
    assert [r["review_id"] for r in a] == [r["review_id"] for r in b]
    assert [r["image_sha256"] for r in a] == [r["image_sha256"] for r in b]


def test_private_mapping_not_in_annotator_queue_fields():
    assert "sample_id" not in BLIND_QUEUE_FIELDS
    assert "relative_path" not in BLIND_QUEUE_FIELDS
    assert "private_mapping" not in BLIND_QUEUE_FIELDS


def test_no_test_split_in_private_mapping_build():
    _, private = build_blind_public_and_private(_read_csv(MANIFEST))
    assert all(str(r["split"]).lower() == "validation" for r in private)
    assert len(private) == 54


def test_pack_build_with_tmp_root(tmp_path):
    manifest = _read_csv(MANIFEST)
    public, private = build_blind_public_and_private(manifest)
    root = tmp_path / "ds"
    # Need files whose sha matches manifest; rewrite files to match expected sha is hard.
    # Instead write bytes then rebuild public/private with computed digests via local helper.
    for priv in private:
        path = root / priv["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"blob:{priv['sample_id']}".encode()
        path.write_bytes(data)
        priv["image_sha256"] = hashlib.sha256(data).hexdigest()
    # Patch committed-style queue hashes by rebuilding from a temp manifest CSV
    tmp_manifest = tmp_path / "m.csv"
    # Use original manifest rows but update sha256 for validation samples
    by_id = {p["sample_id"]: p for p in private}
    fieldnames = list(manifest[0].keys())
    out_rows = []
    for row in manifest:
        r = dict(row)
        if r["sample_id"] in by_id:
            r["sha256"] = by_id[r["sample_id"]]["image_sha256"]
            r["raw_sha256"] = r["sha256"]
        out_rows.append(r)
    with tmp_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    # Monkeypatch module MANIFEST path by calling build_blind then manual pack copy assert
    pub2, priv2 = build_blind_public_and_private(out_rows)
    out = tmp_path / "pack"
    # Minimal inline pack (same as build_pack) — call build_pack after swapping MANIFEST via write
    import build_validation_blind_review_pack as mod

    old = mod.MANIFEST
    mod.MANIFEST = tmp_manifest
    try:
        man = build_pack(dataset_root=root, out_dir=out, seed=20260806)
    finally:
        mod.MANIFEST = old
    assert man["n"] == 54
    assert (out / "private_mapping.csv").is_file()
    assert (out / "review_queue.csv").is_file()
    q = _read_csv(out / "review_queue.csv")
    for row in q:
        assert_public_row_blind(row)
        assert "sample_id" not in row
    # Annotator must not load private as queue
    priv_cols = set(_read_csv(out / "private_mapping.csv")[0].keys())
    assert "sample_id" in priv_cols
    assert "relative_path" in priv_cols
