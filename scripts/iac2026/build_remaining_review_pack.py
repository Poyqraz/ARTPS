"""Build the 306-sample neutral review pack (train+test) for independent_eval_v1_1.

Continuation IDs review_0055..review_0360. The 54 validation reviews from PR #28 stay
immutable and are not re-generated here. Split (including test membership), original /
heuristic label, terrain/folder/Roboflow class, and any model/anomaly score are hidden
from the public queue; they live only in the gitignored private mapping.

Test images are copied for human annotation only. This tool performs NO ARTPS / PaDiM /
PatchCore inference on any split. Fail-closed on missing dataset root or SHA mismatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "iac2026"))

from build_validation_blind_review_pack import (  # noqa: E402
    _read_csv,
    _sha256_file,
    resolve_dataset_root,
)
from validation_blind_review import (  # noqa: E402
    BLIND_QUEUE_FIELDS,
    BLIND_QUEUE_SEED,
    EXPECTED_REMAINING_N,
    PRIVATE_MAPPING_FIELDS,
    REMAINING_ID_OFFSET,
    assert_public_row_blind,
    build_blind_queue_for_rows,
    non_validation_rows,
)

MANIFEST = REPO_ROOT / "reproduction" / "iac2026" / "manifests" / "independent_eval_v1.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "iac2026" / "independent_eval_v1" / "remaining_review_pack"


def build_pack(*, dataset_root: Path, out_dir: Path, seed: int = BLIND_QUEUE_SEED) -> dict:
    manifest_rows = _read_csv(MANIFEST)
    rows = non_validation_rows(manifest_rows)
    if len(rows) != EXPECTED_REMAINING_N:
        raise ValueError(
            f"expected {EXPECTED_REMAINING_N} non-validation rows, got {len(rows)} (fail closed)"
        )
    public, private = build_blind_queue_for_rows(
        rows, seed=seed, id_offset=REMAINING_ID_OFFSET
    )
    for row in public:
        assert_public_row_blind(row)

    review_ids = [r["review_id"] for r in public]
    if review_ids[0] != "review_0055" or review_ids[-1] != "review_0360":
        raise ValueError(f"unexpected id range: {review_ids[0]}..{review_ids[-1]}")
    if len(set(review_ids)) != EXPECTED_REMAINING_N:
        raise ValueError("duplicate review_id in remaining pack")

    images_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir.mkdir(parents=True)

    for pub, priv in zip(public, private):
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
        "protocol_id": "independent_eval_v1_1",
        "phase": "remaining_306_manual_review",
        "seed": seed,
        "n": len(public),
        "id_range": [review_ids[0], review_ids[-1]],
        "dataset_root": str(dataset_root),
        "public_fields": BLIND_QUEUE_FIELDS,
        "private_fields": PRIVATE_MAPPING_FIELDS,
        "split_hidden": True,
        "test_membership_hidden": True,
        "model_scores_visible": False,
        "original_labels_visible": False,
        "no_inference_performed": True,
        "note": "private_mapping.csv must not be shown in annotator UI; test images are for "
        "human annotation only, no ARTPS/PaDiM/PatchCore inference on any split",
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
    print(
        f"Wrote remaining pack n={manifest['n']} "
        f"({manifest['id_range'][0]}..{manifest['id_range'][1]}) -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
