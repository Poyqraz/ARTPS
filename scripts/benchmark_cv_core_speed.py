#!/usr/bin/env python3
"""C07 workstation core-only speed harness (no learned depth / no AE).

Reports the FPS this machine produces. Does **not** claim a match to 28.1.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "iac2026"))

from _common import (  # noqa: E402
    REPO_ROOT,
    copy_config_sidecar,
    environment_snapshot,
    git_dirty,
    load_yaml,
    make_run_id,
    resolve_repo_path,
    write_json,
    write_text,
)
from cv_core_pipeline import core_process_rgb_u8  # noqa: E402


def _collect_frames(
    *,
    images_dir: Optional[Path],
    synthetic_frames: int,
    resolution: int,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    if images_dir is not None and images_dir.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
        for p in paths:
            img = Image.open(p).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
            frames.append(np.asarray(img, dtype=np.uint8))
    if not frames:
        rng = np.random.default_rng(0)
        for i in range(max(1, int(synthetic_frames))):
            frames.append(rng.integers(0, 256, size=(resolution, resolution, 3), dtype=np.uint8))
    return frames


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark OpenCV-only core pipeline (C07 harness).")
    ap.add_argument("--config", default=None, help="YAML config (preferred)")
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--timed", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--allow-learned-depth", action="store_true", help="Forbidden for C07; rejected.")
    ap.add_argument("--allow-autoencoder", action="store_true", help="Forbidden for C07; rejected.")
    args = ap.parse_args(argv)

    cfg: Dict[str, Any] = {}
    config_path: Optional[Path] = None
    if args.config:
        config_path = resolve_repo_path(args.config)
        cfg = load_yaml(config_path)

    resolution = int(args.resolution or cfg.get("input_resolution", 256))
    warmup = int(args.warmup or cfg.get("warmup_count", 30))
    timed_n = int(args.timed or cfg.get("timed_iteration_count", 300))
    batch_size = int(cfg.get("batch_size", 1))
    learned_depth = bool(cfg.get("learned_depth_enabled", False)) or bool(args.allow_learned_depth)
    ae_enabled = bool(cfg.get("autoencoder_enabled", False)) or bool(args.allow_autoencoder)

    if resolution != 256:
        print(f"WARNING: C07 abstract uses 256; this run uses {resolution}", file=sys.stderr)
    if batch_size != 1:
        raise SystemExit("C07 harness requires batch_size=1")
    if learned_depth:
        raise SystemExit("C07 harness forbids learned depth (set learned_depth_enabled: false)")
    if ae_enabled:
        raise SystemExit("C07 harness forbids autoencoder (set autoencoder_enabled: false)")
    if timed_n < 300:
        raise SystemExit("C07 harness requires timed_iteration_count >= 300")
    if warmup < 30:
        raise SystemExit("C07 harness requires warmup_count >= 30")

    images_dir = None
    if args.images_dir or cfg.get("images_dir"):
        images_dir = resolve_repo_path(str(args.images_dir or cfg.get("images_dir")))

    frames = _collect_frames(
        images_dir=images_dir,
        synthetic_frames=int(cfg.get("synthetic_frames", 8)),
        resolution=resolution,
    )

    run_id = args.run_id or make_run_id("c07_core")
    out_dir = resolve_repo_path(cfg.get("output_directory", "results/iac2026/reproduction")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up (not recorded)
    for i in range(warmup):
        core_process_rgb_u8(frames[i % len(frames)])

    stage_core_s: List[float] = []
    stage_resize_s: List[float] = []
    raw_rows: List[Dict[str, Any]] = []

    for i in range(timed_n):
        src = frames[i % len(frames)]
        t_resize0 = time.perf_counter_ns()
        # Frames are already at target resolution; record a no-op-ish resize for stage bookkeeping.
        rgb = cv2.resize(src, (resolution, resolution), interpolation=cv2.INTER_AREA)
        t_resize1 = time.perf_counter_ns()

        t0 = time.perf_counter_ns()
        combined, dets = core_process_rgb_u8(rgb)
        t1 = time.perf_counter_ns()

        resize_s = (t_resize1 - t_resize0) / 1e9
        core_s = (t1 - t0) / 1e9
        stage_resize_s.append(resize_s)
        stage_core_s.append(core_s)
        raw_rows.append(
            {
                "iter": i,
                "resize_s": resize_s,
                "core_processing_s": core_s,
                "n_detections": len(dets),
                "map_mean": float(np.mean(combined)),
            }
        )

    mean_core = float(statistics.fmean(stage_core_s))
    summary = {
        "claim_ids": cfg.get("claim_ids", ["C07"]),
        "input_resolution": resolution,
        "batch_size": batch_size,
        "learned_depth_enabled": False,
        "autoencoder_enabled": False,
        "warmup_count": warmup,
        "timed_iteration_count": timed_n,
        "frame_count": len(frames),
        "mean_core_latency_s": mean_core,
        "median_core_latency_s": float(statistics.median(stage_core_s)),
        "p95_core_latency_s": _percentile(stage_core_s, 95),
        "p99_core_latency_s": _percentile(stage_core_s, 99),
        "min_core_latency_s": float(min(stage_core_s)),
        "max_core_latency_s": float(max(stage_core_s)),
        "std_core_latency_s": float(statistics.pstdev(stage_core_s)) if len(stage_core_s) > 1 else 0.0,
        "fps_from_mean_core": float(1.0 / mean_core) if mean_core > 0 else 0.0,
        "stages": {
            "resize_mean_s": float(statistics.fmean(stage_resize_s)),
            "core_processing_mean_s": mean_core,
        },
        "notes": (
            "Headline FPS is 1/mean(core_processing). "
            "Does not claim equality with accepted-abstract 28.1 FPS."
        ),
    }

    with (out_dir / "timing_raw.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["iter", "resize_s", "core_processing_s", "n_detections", "map_mean"],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    write_json(out_dir / "timing_summary.json", summary)
    write_json(out_dir / "environment.json", environment_snapshot())
    write_json(
        out_dir / "provenance.json",
        {
            "git_head": environment_snapshot()["git_head"],
            "git_dirty": git_dirty(),
            "command": " ".join(sys.argv),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "accepted_abstract_28_1_not_used_as_pass_fail": True,
        },
    )
    if config_path is not None:
        copy_config_sidecar(config_path, out_dir)
    write_text(out_dir / "command.txt", " ".join(sys.argv) + "\n")

    print("=== C07 core-only benchmark ===")
    print(f"resolution        : {resolution}x{resolution}")
    print(f"warmup / timed    : {warmup} / {timed_n}")
    print(f"mean_core_s       : {mean_core:.6f}")
    print(f"fps_from_mean_core: {summary['fps_from_mean_core']:.2f}")
    print(f"artifacts         : {out_dir}")
    print("(No claim of match to accepted-abstract 28.1 FPS.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
