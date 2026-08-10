"""Author-pool RGB-locked non-test 4-panel qualitative ARTPS illustration (not a benchmark)."""

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
from _candidate_support_overlay import (  # noqa: E402
    OVERLAY_VISUALIZATION_VERSION,
    candidate_xywh_scores,
    draw_candidate_support_overlay,
    overlay_geometry_counts,
)
from _figure_typography import (  # noqa: E402
    FIG_DPI,
    apply_manuscript_serif,
    save_manuscript_figure,
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

LOCKED_RELATIVE_PATH = (
    "train/boulder/curiosity_300_MAST_453_jpg.rf.6ecd29659d982741653bbe91b11ef22b.jpg"
)
LOCKED_FILE_SHA256 = "a94b785f4e9cf88fcf07ae7ddabe5b79c53957afe764aac3b14ba3125d7571a2"
SELECTION_ORIGIN = "AUTHOR_1"
AUTHOR_RGB_POOL = {
    "AUTHOR_1": LOCKED_RELATIVE_PATH,
    "AUTHOR_2": "0735MR0031500040403079E01_DXXX.jpg",
    "AUTHOR_3": "Mars_Perseverance_Rover_Sands.png.png",
    "AUTHOR_4": (
        "train/boulder/curiosity_1500_MAST_1736_jpg.rf.a8aeaeffa6df777cc46abce36193bec4.jpg"
    ),
}
SELECTION_RULE = (
    "selected before inference from an author-provided RGB candidate pool using "
    "qualitative scene-composition and domain criteria; no ARTPS output, score, "
    "or detection count used for selection"
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


def lock_sample(root: Path) -> dict[str, str]:
    rel = LOCKED_RELATIVE_PATH.replace("\\", "/").lstrip("/")
    if rel.lower().startswith("test/"):
        raise RuntimeError(f"locked path is under test/: {rel}")
    path = root / rel
    if not path.is_file():
        raise RuntimeError(f"locked source missing: {path}")
    row: dict[str, str] = {}
    if MANIFEST.is_file():
        for cand in csv.DictReader(MANIFEST.open(encoding="utf-8")):
            mrel = (cand.get("relative_path") or "").replace("\\", "/").lstrip("/")
            if mrel == rel:
                row = cand
                break
    split = (row.get("split") or rel.split("/", 1)[0]).strip()
    if split.lower() == "test":
        raise RuntimeError(f"locked source is test split: {rel}")
    included = (row.get("inclusion_status") or "").strip().lower() == "included"
    sample_id = (row.get("sample_id") or "").strip() or f"author_pool_{SELECTION_ORIGIN}_{path.name}"
    return {
        "sample_id": sample_id,
        "split": split,
        "relative_path": rel,
        "product_id": (row.get("product_id") or "curiosity_300_mast_453").strip(),
        "manifest_sha256": (row.get("sha256") or "").strip(),
        "in_independent_eval_v1_included": included,
        "abs_path": str(path.resolve()),
    }


def _to_u8(rgb_float: np.ndarray) -> np.ndarray:
    return np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()


def main() -> int:
    _assert_test_closed()
    root = _dataset_root()
    sample = lock_sample(root)
    split_l = sample["split"].lower()
    if split_l == "test":
        raise RuntimeError("selected sample is test; abort")
    file_sha = _sha256_file(Path(sample["abs_path"]))
    if file_sha != LOCKED_FILE_SHA256:
        raise RuntimeError(f"locked source sha256 mismatch: {file_sha}")
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
    overlay = draw_candidate_support_overlay(rgb_disp, scored, combined_map, (mh, mw))
    geom = overlay_geometry_counts(scored)
    depth_n = depth_map.astype(np.float32)
    depth_n = (depth_n - depth_n.min()) / (float(depth_n.max() - depth_n.min()) + 1e-8)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    apply_manuscript_serif()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), dpi=FIG_DPI)
    panels = [
        (axes[0, 0], rgb_disp, "a) RGB input", None),
        (axes[0, 1], combined_map, "b) Post-suppression combined map", "inferno"),
        (axes[1, 0], depth_n, "c) Relative depth (near to far)", "viridis"),
        (axes[1, 1], overlay, "d) Candidate-support overlay", None),
    ]
    for ax, img, title, cmap in panels:
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=0.35)
    save_manuscript_figure(fig, OUT_PNG, dpi=FIG_DPI)

    meta = {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "relative_path": sample["relative_path"],
        "product_id": sample["product_id"],
        "abs_path": sample["abs_path"],
        "file_sha256": file_sha,
        "selection_rule": SELECTION_RULE,
        "selection_origin": SELECTION_ORIGIN,
        "author_rgb_pool": AUTHOR_RGB_POOL,
        "source_selection_locked": True,
        "in_independent_eval_v1_included": sample["in_independent_eval_v1_included"],
        "mission": "Curiosity",
        "instrument": "Mastcam",
        "test_used": False,
        "score_blind_selection": True,
        "model_output_used_for_selection": False,
        "score_based_cherry_picking": False,
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
        "candidates": candidate_xywh_scores(scored),
        "overlay_visualization_version": OVERLAY_VISUALIZATION_VERSION,
        "support_geometry_source": (
            "proposal hysteresis/CC contour persisted as visualization-only metadata; "
            "drawn as low-alpha footprint, not a polyline silhouette; no new map threshold"
        ),
        "anchor_definition": (
            "argmax of post-suppression combined_map inside support contour, else ROI; "
            "peak_xy if present"
        ),
        "fallback_behavior": "open-corner ROI + anchor when no proposal CC survives",
        "visualization_only": True,
        "candidate_scores_changed": False,
        "validity_decisions_changed": False,
        "image_scores_changed": False,
        "quantitative_experiment": False,
        **geom,
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
