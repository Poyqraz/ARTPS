"""Build local blind-review pack (images + public queue + private mapping). Fail-closed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "iac2026"))

from validation_blind_review import (  # noqa: E402
    BLIND_QUEUE_FIELDS,
    BLIND_QUEUE_SEED,
    PRIVATE_MAPPING_FIELDS,
    assert_public_row_blind,
    build_blind_public_and_private,
)

MANIFEST = REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "iac2026" / "independent_eval_v1" / "blind_review_pack"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_dataset_root(explicit: str | None = None) -> Path:
    raw = explicit or os.environ.get("ARTPS_DATASET_ROOT")
    if not raw or not str(raw).strip():
        raise ValueError(
            "ARTPS_DATASET_ROOT unset: refuse blind pack build (fail closed)"
        )
    root = Path(raw)
    if not root.is_dir():
        raise ValueError(f"ARTPS_DATASET_ROOT is not a directory: {root}")
    return root.resolve()


def build_pack(
    *,
    dataset_root: Path,
    out_dir: Path,
    seed: int = BLIND_QUEUE_SEED,
) -> dict:
    manifest_rows = _read_csv(MANIFEST)
    public, private = build_blind_public_and_private(manifest_rows, seed=seed)
    if len(public) != 54:
        raise ValueError(f"expected 54 validation rows, got {len(public)}")
    for row in public:
        assert_public_row_blind(row)
        if any(s in str(row.get("split", "")).lower() for s in ("test",)):
            raise ValueError("test split leaked into public queue")

    images_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True)

    for pub, priv in zip(public, private):
        if str(priv["split"]).strip().lower() == "test":
            raise ValueError("test image included in blind pack")
        src = dataset_root / priv["relative_path"]
        if not src.is_file():
            raise ValueError(f"missing source image: {src}")
        digest = _sha256_file(src)
        expected = priv["image_sha256"]
        if expected and digest != expected:
            raise ValueError(
                f"sha256 mismatch for {priv['sample_id']}: got {digest}, expected {expected}"
            )
        dest = images_dir / pub["neutral_filename"]
        shutil.copyfile(src, dest)
        if _sha256_file(dest) != digest:
            raise ValueError(f"byte-copy sha mismatch: {dest}")

    with (out_dir / "review_queue.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BLIND_QUEUE_FIELDS)
        w.writeheader()
        w.writerows(public)

    with (out_dir / "private_mapping.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRIVATE_MAPPING_FIELDS)
        w.writeheader()
        w.writerows(private)

    pack_manifest = {
        "protocol_id": "independent_eval_v1",
        "seed": seed,
        "n": len(public),
        "dataset_root": str(dataset_root),
        "public_fields": BLIND_QUEUE_FIELDS,
        "private_fields": PRIVATE_MAPPING_FIELDS,
        "note": "private_mapping.csv must not be shown in annotator UI",
        "image_sha256_list": [r["image_sha256"] for r in public],
    }
    (out_dir / "pack_manifest.json").write_text(
        json.dumps(pack_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return pack_manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=BLIND_QUEUE_SEED)
    args = p.parse_args(argv)
    root = resolve_dataset_root(args.dataset_root)
    manifest = build_pack(dataset_root=root, out_dir=args.out_dir, seed=args.seed)
    print(f"Wrote pack n={manifest['n']} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
