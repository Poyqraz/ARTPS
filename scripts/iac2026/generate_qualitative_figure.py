"""Deterministic non-test 4-panel qualitative ARTPS illustration (not a benchmark)."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.artps_detection_core import (  # noqa: E402
    _compute_protrusion_map,
    _score_object_detections,
    compute_combined_anomaly_map,
    set_runtime_params,
)
from src.artps_inference import (  # noqa: E402
    FrozenARTPSConfig,
    _ae_forward,
    _depth_for_fusion,
    _known_value_score,
    _preprocess_image,
    load_frozen_artps_profile,
)

MANIFEST = REPO / "reproduction/iac2026/manifests/independent_eval_v1.csv"
FREEZE = REPO / "reproduction/iac2026/test_freeze/TEST_OPEN_STATUS.yaml"
SCOPE = REPO / "reproduction/iac2026/test_freeze/FINAL_TEST_SCOPE.yaml"
CONFIG_YAML = REPO / "reproduction/iac2026/configs/independent_eval_artps_full_frozen_mars.yaml"
OUT_PNG = REPO / "paper/iac2026/figures/fig_qualitative_artps.png"
OUT_META = REPO / "paper/iac2026/figures/fig_qualitative_artps.meta.json"

AE_SHA = "8186bbe6be424dd212d5d4a93b1ae36b80939552519706b4a8680c5d05e995f2"
DPT_SHA = "2f21e586477d90cb9624c7eef5df7891edca49a1c4795ee2cb631fd4daa6ca69"
CLF_SHA = "83f6c63eeef6ede9ce7e2fed47acf0d594ec1f957684ae357f23a6f0dd491457"

SELECTION_RULE = (
    "included rows from independent_eval_v1.csv; drop split==test or relative_path "
    "under test/; sort by sample_id UTF-8 lexicographic; first existing file under "
    "ARTPS_DATASET_ROOT; no score lookup before selection"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_root() -> Path:
    raw = os.environ.get("ARTPS_DATASET_ROOT", "").strip()
    if not raw:
        raise RuntimeError("ARTPS_DATASET_ROOT is unset")
    root = Path(raw).resolve()
    if not root.is_dir():
        raise RuntimeError(f"ARTPS_DATASET_ROOT is not a directory: {root}")
    return root


def _assert_test_closed() -> None:
    for path in (FREEZE, SCOPE):
        status = yaml.safe_load(path.read_text(encoding="utf-8"))
        if status.get("test_opened") is not False:
            raise RuntimeError(f"test split is not closed: {path}")
        if path == SCOPE and status.get("final_test_authorized") is not False:
            raise RuntimeError("final_test_authorized is not false")


def select_sample(root: Path) -> dict[str, str]:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    cands: list[dict[str, str]] = []
    for row in rows:
        if (row.get("inclusion_status") or "").strip().lower() != "included":
            continue
        split = (row.get("split") or "").strip().lower()
        rel = (row.get("relative_path") or "").replace("\\", "/").lstrip("/")
        if split == "test" or rel.lower().startswith("test/"):
            continue
        cands.append(row)
    cands.sort(key=lambda r: r["sample_id"])
    for row in cands:
        rel = row["relative_path"].replace("\\", "/").lstrip("/")
        path = root / rel
        if path.is_file():
            return {
                "sample_id": row["sample_id"],
                "split": row["split"].strip(),
                "relative_path": rel,
                "product_id": row.get("product_id", ""),
                "manifest_sha256": row.get("sha256", ""),
                "abs_path": str(path.resolve()),
            }
    raise RuntimeError("no non-test included manifest image exists under ARTPS_DATASET_ROOT")


def _to_u8(rgb_float: np.ndarray) -> np.ndarray:
    return np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)


def _overlay_boxes(rgb_u8: np.ndarray, detections: list[dict], map_hw: tuple[int, int]) -> np.ndarray:
    out = rgb_u8.copy()
    mh, mw = map_hw
    h, w = out.shape[:2]
    sx = w / float(mw)
    sy = h / float(mh)
    for det in detections:
        x1 = int(round(float(det["x"]) * sx))
        y1 = int(round(float(det["y"]) * sy))
        x2 = int(round((float(det["x"]) + float(det["w"])) * sx))
        y2 = int(round((float(det["y"]) + float(det["h"])) * sy))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 80), 2)
    return out


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()


def main() -> int:
    _assert_test_closed()
    root = _dataset_root()
    sample = select_sample(root)
    split_l = sample["split"].lower()
    if split_l == "test":
        raise RuntimeError("selected sample is test; abort")
    file_sha = _sha256_file(Path(sample["abs_path"]))
    if sample["manifest_sha256"] and file_sha != sample["manifest_sha256"]:
        raise RuntimeError(
            f"source sha256 mismatch: manifest={sample['manifest_sha256']} file={file_sha}"
        )

    cfg = FrozenARTPSConfig(
        preprocessing_profile="mars_enhancement_v1",
        use_amp=False,
        enable_classifier=True,
        checkpoint_sha256={"ae": AE_SHA, "dpt": DPT_SHA, "classifier": CLF_SHA},
    )
    bundle = load_frozen_artps_profile(cfg)
    set_runtime_params(cfg.detection_params())

    pil = _preprocess_image(Path(sample["abs_path"]), cfg.preprocessing_profile)
    mse, original, reconstructed, latent = _ae_forward(
        bundle.autoencoder,
        pil,
        bundle.device,
        ae_resize=cfg.ae_resize,
        use_amp=False,
    )
    known_value = _known_value_score(bundle, original, latent)
    depth_map = _depth_for_fusion(bundle, pil)
    protrusion_map = _compute_protrusion_map(depth_map)
    combined_map, detections, _diag = compute_combined_anomaly_map(
        original,
        reconstructed,
        depth_map,
        hyst_high_pct=cfg.hyst_high,
        hyst_low_pct=cfg.hyst_low,
        nms_iou=cfg.nms_iou,
        top_k=cfg.top_k,
        w_recon=cfg.w_recon,
        w_depth=cfg.w_depth,
        w_texture=cfg.w_texture,
        edge_reinforce=cfg.edge_reinf,
    )
    scored = _score_object_detections(
        detections,
        original_rgb_float=original,
        autoencoder=bundle.autoencoder,
        device=bundle.device,
        combined_map=combined_map,
        depth_map=depth_map,
        protrusion_map=protrusion_map,
        padim_map=None,
        patchcore_map=None,
        global_known_value=known_value,
    )

    rgb_in = np.array(pil.convert("RGB"), dtype=np.uint8)
    mh, mw = combined_map.shape[:2]
    rgb_disp = cv2.resize(rgb_in, (mw, mh), interpolation=cv2.INTER_AREA)
    overlay = _overlay_boxes(rgb_disp, scored, (mh, mw))
    depth_n = depth_map.astype(np.float32)
    depth_n = (depth_n - depth_n.min()) / (float(depth_n.max() - depth_n.min()) + 1e-8)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), dpi=200)
    panels = [
        (axes[0, 0], rgb_disp, "a) RGB input", None),
        (axes[0, 1], combined_map, "b) Post-suppression combined map", "inferno"),
        (axes[1, 0], depth_n, "c) Relative depth (near to far)", "viridis"),
        (axes[1, 1], overlay, "d) Valid-candidate overlay", None),
    ]
    for ax, img, title, cmap in panels:
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta = {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "relative_path": sample["relative_path"],
        "product_id": sample["product_id"],
        "abs_path": sample["abs_path"],
        "file_sha256": file_sha,
        "selection_rule": SELECTION_RULE,
        "test_used": False,
        "score_blind_selection": True,
        "git_sha_at_generation": _git_sha(),
        "config_yaml": str(CONFIG_YAML.relative_to(REPO)).replace("\\", "/"),
        "config_id": "artps_full_frozen_mars_clf_on_v1",
        "preprocessing_profile": "mars_enhancement_v1",
        "enable_realesrgan": False,
        "priority_buffer": False,
        "curiosity_ranking_applied": False,
        "diversity_penalty_applied": False,
        "image_score_aggregation": "max_valid_candidate_after_masks",
        "checkpoints": {
            "ae": {"path": cfg.ae_path, "sha256": bundle.checkpoint_hashes.get("ae", "")},
            "dpt": {
                "path": "raw_models/dpt_large_384.pt",
                "sha256": bundle.checkpoint_hashes.get("dpt", ""),
            },
            "classifier": {
                "path": cfg.classifier_path,
                "sha256": bundle.checkpoint_hashes.get("classifier", ""),
            },
        },
        "device": str(bundle.device),
        "n_raw_detections": len(detections),
        "n_valid_candidates": len(scored),
        "output_png": str(OUT_PNG.relative_to(REPO)).replace("\\", "/"),
        "ae_mse_not_a_manuscript_metric": float(mse),
        "command": (
            'set ARTPS_DATASET_ROOT=<repo>/mars_images && '
            "python scripts/iac2026/generate_qualitative_figure.py"
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
