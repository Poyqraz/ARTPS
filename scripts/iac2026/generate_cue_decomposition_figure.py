"""Fig. 3: frozen cue decomposition on the exact Fig. 2 non-test sample."""

from __future__ import annotations

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

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.axes_grid1 import ImageGrid

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_SCRIPTS = str(REPO / "scripts" / "iac2026")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from src.artps_detection_core import compute_combined_anomaly_map, set_runtime_params  # noqa: E402
from src.artps_inference import (  # noqa: E402
    FrozenARTPSConfig,
    _ae_forward,
    _depth_for_fusion,
    _preprocess_image,
    load_frozen_artps_profile,
)
from _figure_typography import (  # noqa: E402
    FIG_DPI,
    apply_manuscript_serif,
    save_manuscript_figure,
)

FIG2_META = REPO / "paper/iac2026/figures/fig_qualitative_artps.meta.json"
FREEZE = REPO / "reproduction/iac2026/test_freeze/TEST_OPEN_STATUS.yaml"
SCOPE = REPO / "reproduction/iac2026/test_freeze/FINAL_TEST_SCOPE.yaml"
CONFIG_YAML = REPO / "reproduction/iac2026/configs/independent_eval_artps_full_frozen_mars.yaml"
OUT_PNG = REPO / "paper/iac2026/figures/fig_cue_decomposition_artps.png"
OUT_META = REPO / "paper/iac2026/figures/fig_cue_decomposition_artps.meta.json"

AE_SHA = "8186bbe6be424dd212d5d4a93b1ae36b80939552519706b4a8680c5d05e995f2"
DPT_SHA = "2f21e586477d90cb9624c7eef5df7891edca49a1c4795ee2cb631fd4daa6ca69"
CLF_SHA = "83f6c63eeef6ede9ce7e2fed47acf0d594ec1f957684ae357f23a6f0dd491457"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_test_closed() -> None:
    for path in (FREEZE, SCOPE):
        status = yaml.safe_load(path.read_text(encoding="utf-8"))
        if status.get("test_opened") is not False:
            raise RuntimeError(f"test split is not closed: {path}")
        if path == SCOPE and status.get("final_test_authorized") is not False:
            raise RuntimeError("final_test_authorized is not false")


def _dataset_root() -> Path:
    raw = os.environ.get("ARTPS_DATASET_ROOT", "").strip()
    if not raw:
        raise RuntimeError("ARTPS_DATASET_ROOT is unset")
    root = Path(raw).resolve()
    if not root.is_dir():
        raise RuntimeError(f"ARTPS_DATASET_ROOT is not a directory: {root}")
    return root


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()


def _display_minmax(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def main() -> int:
    _assert_test_closed()
    fig2 = json.loads(FIG2_META.read_text(encoding="utf-8"))
    if str(fig2.get("split", "")).lower() == "test":
        raise RuntimeError("Fig. 2 sample is test; abort")
    expected_sha = str(fig2.get("file_sha256") or "")
    if not expected_sha:
        raise RuntimeError("Fig. 2 meta missing file_sha256")

    root = _dataset_root()
    rel = str(fig2["relative_path"]).replace("\\", "/").lstrip("/")
    if rel.lower().startswith("test/"):
        raise RuntimeError(f"Fig. 2 relative_path is under test/: {rel}")
    src = root / rel
    if not src.is_file():
        raise RuntimeError(f"Fig. 2 source missing under ARTPS_DATASET_ROOT: {src}")
    file_sha = _sha256_file(src)
    if file_sha != expected_sha:
        raise RuntimeError(f"source sha256 mismatch vs Fig. 2 meta: {file_sha}")

    cfg = FrozenARTPSConfig(
        preprocessing_profile="mars_enhancement_v1",
        use_amp=False,
        enable_classifier=True,
        checkpoint_sha256={"ae": AE_SHA, "dpt": DPT_SHA, "classifier": CLF_SHA},
    )
    bundle = load_frozen_artps_profile(cfg)
    set_runtime_params(cfg.detection_params())

    pil = _preprocess_image(src, cfg.preprocessing_profile)
    _mse, original, reconstructed, _latent = _ae_forward(
        bundle.autoencoder,
        pil,
        bundle.device,
        ae_resize=cfg.ae_resize,
        use_amp=False,
    )
    depth_map = _depth_for_fusion(bundle, pil)
    _combined, _dets, diag = compute_combined_anomaly_map(
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
    for key in ("recon_diff_n", "depth_edge_n", "texture_term", "raw_combined_pre_mask"):
        if key not in diag:
            raise RuntimeError(f"frozen diagnostics missing cue map: {key}")

    recon = _display_minmax(np.asarray(diag["recon_diff_n"]))
    depth_edge = _display_minmax(np.asarray(diag["depth_edge_n"]))
    texture = _display_minmax(np.asarray(diag["texture_term"]))
    fused = _display_minmax(np.asarray(diag["raw_combined_pre_mask"]))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    apply_manuscript_serif()
    # ImageGrid keeps equal-aspect panels adjacent; subplots_adjust+imshow
    # left ~148–240 px mid-column dead space on a wide figsize.
    fig = plt.figure(figsize=(7.2, 5.2), dpi=FIG_DPI)
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=(0.08, 0.24))
    panels = [
        (grid[0], recon, "a) Reconstruction residual", "inferno"),
        (grid[1], depth_edge, "b) Relative-depth edge", "inferno"),
        (grid[2], texture, "c) Texture / local contrast", "inferno"),
        (grid[3], fused, "d) Pre-suppression fused map", "inferno"),
    ]
    for ax, img, title, cmap in panels:
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    save_manuscript_figure(fig, OUT_PNG, dpi=FIG_DPI)

    meta = {
        "sample_id": fig2["sample_id"],
        "product_id": fig2.get("product_id", ""),
        "split": fig2["split"],
        "relative_path": rel,
        "file_sha256": file_sha,
        "source_selection_inherited_from_fig2": True,
        "new_sample_selection": False,
        "test_used": False,
        "score_blind_selection": True,
        "score_based_cherry_picking": False,
        "quantitative_experiment": False,
        "classifier_in_fused_map": False,
        "git_sha_at_generation": _git_sha(),
        "config_yaml": str(CONFIG_YAML.relative_to(REPO)).replace("\\", "/"),
        "config_id": "artps_full_frozen_mars_clf_on_v1",
        "preprocessing_profile": "mars_enhancement_v1",
        "enable_realesrgan": False,
        "priority_buffer": False,
        "curiosity_ranking_applied": False,
        "diversity_penalty_applied": False,
        "cue_implementation": "src/artps_detection_core.py:compute_combined_anomaly_map",
        "inference_normalization": "_normalize_map percentile 2-98 per cue before weighting",
        "visualization_normalization": "per-panel min-max for display only",
        "frozen_fusion_weights": {
            "w_recon": float(diag["w_recon"]),
            "w_depth": float(diag["w_depth"]),
            "w_texture": float(diag["w_texture"]),
            "w_lap": float(diag["w_lap"]),
            "w_detail": float(diag["w_detail"]),
        },
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
            "set ARTPS_DATASET_ROOT=<repo>/mars_images then "
            "python scripts/iac2026/generate_cue_decomposition_figure.py"
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
