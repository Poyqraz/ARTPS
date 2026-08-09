"""Author-pool close vs distant qualitative ARTPS illustration (not a benchmark)."""

from __future__ import annotations

import argparse
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
    candidate_xywh_scores,
    draw_candidate_support_overlay,
    overlay_geometry_counts,
)

MANIFEST = REPO / "reproduction/iac2026/manifests/independent_eval_v1.csv"
FREEZE = REPO / "reproduction/iac2026/test_freeze/TEST_OPEN_STATUS.yaml"
SCOPE = REPO / "reproduction/iac2026/test_freeze/FINAL_TEST_SCOPE.yaml"
CONFIG_YAML = REPO / "reproduction/iac2026/configs/independent_eval_artps_full_frozen_mars.yaml"
OUT_PNG = REPO / "paper/iac2026/figures/fig_close_far_qualitative_artps.png"
OUT_META = REPO / "paper/iac2026/figures/fig_close_far_qualitative_artps.meta.json"
PREVIEW_DIR = REPO / "tmp" / "close_far_preview"

AE_SHA = "8186bbe6be424dd212d5d4a93b1ae36b80939552519706b4a8680c5d05e995f2"
DPT_SHA = "2f21e586477d90cb9624c7eef5df7891edca49a1c4795ee2cb631fd4daa6ca69"
CLF_SHA = "83f6c63eeef6ede9ce7e2fed47acf0d594ec1f957684ae357f23a6f0dd491457"

ALLOWED_POOL: dict[str, str] = {
    "close1": (
        "train/hills_or_ridge/"
        "curiosity_1100_MAST_938_jpg.rf.7417a3036ec4af81b3b9d4305c05eee3.jpg"
    ),
    "close2": (
        "train/boulder/"
        "percy_sol1450_MCZ_RIGHT_9_jpg.rf.f390f8c84becbe615a34db73d9f2610e.jpg"
    ),
    "far1": (
        "train/flat_terrain/"
        "curiosity_1100_MAST_827_jpg.rf.fd10bd35d413cba7432b79ab8433e9b6.jpg"
    ),
    "far2": (
        "train/rocky/"
        "curiosity_1100_MAST_817_jpg.rf.7d755ad9d3fcbac273a3dfffdc0b3c40.jpg"
    ),
    "far3": (
        "train/rover/"
        "percy_sol150_NAVCAM_LEFT_8_jpg.rf.5d964d0db273d6db4a7054ec8516c688.jpg"
    ),
}

# Filled after visual preview of all five author-pool candidates (not score-max).
SELECTED_CLOSE = "close1"
SELECTED_FAR = "far2"

SELECTION_RULE = (
    "exactly one close and one far from a fixed author-provided five-image pool; "
    "visual clarity / interpretability / clean post-suppression map / readable overlay; "
    "no score-maximization; test split unused"
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


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()


def _manifest_index() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not MANIFEST.is_file():
        return out
    for row in csv.DictReader(MANIFEST.open(encoding="utf-8")):
        rel = (row.get("relative_path") or "").replace("\\", "/").lstrip("/")
        if rel:
            out[rel] = row
    return out


def _resolve_candidate(key: str, root: Path, index: dict[str, dict[str, str]]) -> dict:
    if key not in ALLOWED_POOL:
        raise RuntimeError(f"candidate key not in author pool: {key}")
    rel = ALLOWED_POOL[key]
    if rel.replace("\\", "/").lower().startswith("test/"):
        raise RuntimeError(f"test path forbidden: {rel}")
    path = root / rel
    if not path.is_file():
        raise RuntimeError(f"missing author-pool image: {rel}")
    row = index.get(rel, {})
    split = (row.get("split") or "").strip() or rel.split("/", 1)[0]
    if split.lower() == "test":
        raise RuntimeError(f"test split forbidden: {rel}")
    sample_id = (row.get("sample_id") or "").strip() or f"author_pool_{key}_{path.name}"
    return {
        "pool_key": key,
        "sample_id": sample_id,
        "split": split,
        "relative_path": rel,
        "product_id": (row.get("product_id") or "").strip(),
        "manifest_sha256": (row.get("sha256") or "").strip(),
        "in_independent_eval_v1_included": (
            (row.get("inclusion_status") or "").strip().lower() == "included"
        ),
        "abs_path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
    }


def _run_frozen(cfg: FrozenARTPSConfig, bundle, abs_path: str) -> dict:
    pil = _preprocess_image(Path(abs_path), cfg.preprocessing_profile)
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
    return {
        "rgb": rgb_disp,
        "combined_map": combined_map,
        "overlay": overlay,
        "n_raw_detections": len(detections),
        "n_valid_candidates": len(scored),
        "candidates": candidate_xywh_scores(scored),
        "overlay_geometry": geom,
        "ae_mse_not_a_manuscript_metric": float(mse),
    }


def _save_row(path: Path, rgb, combined, overlay, titles: tuple[str, str, str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.05), dpi=180)
    panels = [
        (axes[0], rgb, titles[0], None),
        (axes[1], combined, titles[1], "inferno"),
        (axes[2], overlay, titles[2], None),
    ]
    for ax, img, title, cmap in panels:
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_profile():
    cfg = FrozenARTPSConfig(
        preprocessing_profile="mars_enhancement_v1",
        use_amp=False,
        enable_classifier=True,
        checkpoint_sha256={"ae": AE_SHA, "dpt": DPT_SHA, "classifier": CLF_SHA},
    )
    bundle = load_frozen_artps_profile(cfg)
    set_runtime_params(cfg.detection_params())
    return cfg, bundle


def preview_all() -> int:
    _assert_test_closed()
    root = _dataset_root()
    index = _manifest_index()
    cfg, bundle = _load_profile()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for key in ALLOWED_POOL:
        rec = _resolve_candidate(key, root, index)
        out = _run_frozen(cfg, bundle, rec["abs_path"])
        png = PREVIEW_DIR / f"{key}.png"
        _save_row(
            png,
            out["rgb"],
            out["combined_map"],
            out["overlay"],
            (
                f"{key} RGB",
                f"{key} post-suppression map",
                f"{key} valid overlay n={out['n_valid_candidates']}",
            ),
        )
        rec.update(
            {
                "n_raw_detections": out["n_raw_detections"],
                "n_valid_candidates": out["n_valid_candidates"],
                "preview_png": str(png),
            }
        )
        rec.pop("abs_path", None)
        summary.append(rec)
        print(json.dumps(rec, indent=2))
    (PREVIEW_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def generate_figure() -> int:
    _assert_test_closed()
    if SELECTED_CLOSE not in ("close1", "close2") or SELECTED_FAR not in ("far1", "far2", "far3"):
        raise RuntimeError("selected keys must be one close and one far from the author pool")
    root = _dataset_root()
    index = _manifest_index()
    close = _resolve_candidate(SELECTED_CLOSE, root, index)
    far = _resolve_candidate(SELECTED_FAR, root, index)
    if close["manifest_sha256"] and close["file_sha256"] != close["manifest_sha256"]:
        raise RuntimeError(f"close sha256 mismatch: {close['relative_path']}")
    if far["manifest_sha256"] and far["file_sha256"] != far["manifest_sha256"]:
        raise RuntimeError(f"far sha256 mismatch: {far['relative_path']}")

    cfg, bundle = _load_profile()
    close_out = _run_frozen(cfg, bundle, close["abs_path"])
    far_out = _run_frozen(cfg, bundle, far["abs_path"])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.4), dpi=200)
    panels = [
        (axes[0, 0], close_out["rgb"], "a) Close-range RGB", None),
        (axes[0, 1], close_out["combined_map"], "b) Post-suppression combined map", "inferno"),
        (axes[0, 2], close_out["overlay"], "c) Candidate-support overlay", None),
        (axes[1, 0], far_out["rgb"], "d) Distant-scene RGB", None),
        (axes[1, 1], far_out["combined_map"], "e) Post-suppression combined map", "inferno"),
        (axes[1, 2], far_out["overlay"], "f) Candidate-support overlay", None),
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
        "author_provided_pool_only": True,
        "agent_selected_outside_pool": False,
        "quantitative_experiment": False,
        "test_used": False,
        "score_maximization_cherrypick": False,
        "qualitative_selection_rationale": (
            "Close 1 + Far 2: Curiosity Mastcam pair with readable post-suppression maps "
            "and candidate-support overlays. Close 2 rejected for Perseverance domain "
            "mismatch and overlay missing the two most salient near rocks. Far 1 rejected "
            "for a surviving top-edge frame artefact in the valid overlay. Far 3 rejected: "
            "rover hardware still dominates RGB and the map is sparser than Far 2."
        ),
        "rejected": {
            "close2": (
                "Perseverance MCZ domain mismatch vs manuscript Curiosity-Mastcam framing; "
                "overlay misses the two most salient near rocks"
            ),
            "far1": "valid overlay includes a full-width top-edge frame artefact",
            "far3": "rover hardware remains visually dominant in RGB; not cleaner than Far 1/2",
        },
        "selection_rule": SELECTION_RULE,
        "selected_close_key": SELECTED_CLOSE,
        "selected_far_key": SELECTED_FAR,
        "considered_pool": ALLOWED_POOL,
        "close": {
            "pool_key": close["pool_key"],
            "sample_id": close["sample_id"],
            "split": close["split"],
            "relative_path": close["relative_path"],
            "product_id": close["product_id"],
            "file_sha256": close["file_sha256"],
            "in_independent_eval_v1_included": close["in_independent_eval_v1_included"],
            "n_raw_detections": close_out["n_raw_detections"],
            "n_valid_candidates": close_out["n_valid_candidates"],
            "candidates": close_out["candidates"],
            **close_out["overlay_geometry"],
        },
        "far": {
            "pool_key": far["pool_key"],
            "sample_id": far["sample_id"],
            "split": far["split"],
            "relative_path": far["relative_path"],
            "product_id": far["product_id"],
            "file_sha256": far["file_sha256"],
            "in_independent_eval_v1_included": far["in_independent_eval_v1_included"],
            "n_raw_detections": far_out["n_raw_detections"],
            "n_valid_candidates": far_out["n_valid_candidates"],
            "candidates": far_out["candidates"],
            **far_out["overlay_geometry"],
        },
        "overlay_visualization_version": "candidate_support_v1",
        "support_geometry_source": (
            "proposal hysteresis/CC contour persisted as visualization-only metadata; "
            "no new map threshold"
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
        "output_png": str(OUT_PNG.relative_to(REPO)).replace("\\", "/"),
        "command": (
            "set ARTPS_DATASET_ROOT=<repo>/mars_images && "
            "python scripts/iac2026/generate_close_far_qualitative_figure.py"
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-all", action="store_true")
    args = parser.parse_args()
    if args.preview_all:
        return preview_all()
    return generate_figure()


if __name__ == "__main__":
    raise SystemExit(main())
