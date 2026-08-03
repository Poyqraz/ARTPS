#!/usr/bin/env python3
"""C07 workstation speed harness — historical_exact vs current_production profiles."""
from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    environment_snapshot,
    git_dirty,
    git_head,
    load_yaml,
    make_run_id,
    resolve_repo_path,
    sha256_file,
    write_json,
    write_run_bundle,
)
from _config import ConfigValidationError, load_timing_config  # noqa: E402
from cv_core_pipeline import (  # noqa: E402
    PIPELINE_ID,
    SOURCE_COMMIT,
    implementation_hash,
    process_frame_current_production,
    process_frame_historical,
)


def _extended_env() -> Dict[str, Any]:
    env = environment_snapshot()
    try:
        import psutil  # type: ignore

        ram = psutil.virtual_memory().total
        phys = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except Exception:
        ram = None
        phys = None
        logical = os.cpu_count()
    torch_info = None
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception:
        pass
    env.update(
        {
            "cv2_num_threads": cv2.getNumThreads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "cpu_model": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
            "logical_cores": logical,
            "physical_cores": phys,
            "ram_bytes": ram,
            "os": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch_info,
        }
    )
    return env


def _collect_synthetic(resolution: int, n: int) -> List[np.ndarray]:
    rng = np.random.default_rng(0)
    return [
        rng.integers(0, 256, size=(resolution, resolution, 3), dtype=np.uint8)
        for _ in range(max(1, n))
    ]


def _collect_from_dir(images_dir: Path, resolution: int) -> List[np.ndarray]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    frames: List[np.ndarray] = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        frames.append(np.asarray(img, dtype=np.uint8))
    return frames


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C07 core speed harness")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--software-verification", action="store_true")
    ap.add_argument("--images_dir", default=None)
    args = ap.parse_args(argv)

    try:
        cfg = load_timing_config(args.config)
    except ConfigValidationError as exc:
        print(f"CONFIG VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2

    config_path = resolve_repo_path(args.config)
    evidence_mode = str(cfg["evidence_mode"])
    profile = str(cfg["profile"])
    resolution = int(cfg["input_resolution"])
    warmup = int(cfg["warmup_count"])
    timed_n = int(cfg["timed_iteration_count"])
    batch_size = int(cfg["batch_size"])
    learned_depth = bool(cfg["learned_depth_enabled"])
    ae = bool(cfg["autoencoder_enabled"])
    allow_dirty = bool(cfg.get("allow_dirty_git", False))

    if evidence_mode == "real_evidence" and args.software_verification:
        print("real_evidence forbids --software-verification", file=sys.stderr)
        return 2
    if evidence_mode == "software_verification" and not args.software_verification:
        print("software_verification requires --software-verification", file=sys.stderr)
        return 2

    if resolution != 256:
        print("C07 requires input_resolution=256", file=sys.stderr)
        return 2
    if batch_size != 1:
        print("C07 requires batch_size=1", file=sys.stderr)
        return 2
    if learned_depth or ae:
        print("C07 forbids learned depth / autoencoder", file=sys.stderr)
        return 2
    if warmup < 30 or timed_n < 300:
        print("C07 requires warmup>=30 and timed>=300", file=sys.stderr)
        return 2
    if evidence_mode == "real_evidence":
        if git_dirty() and not allow_dirty:
            print("real_evidence: dirty git rejected", file=sys.stderr)
            return 2
        if allow_dirty:
            print("real_evidence should set allow_dirty_git=false", file=sys.stderr)
            return 2

    images_dir = None
    if args.images_dir or cfg.get("images_dir"):
        images_dir = resolve_repo_path(str(args.images_dir or cfg.get("images_dir")))

    input_source = "synthetic"
    frames: List[np.ndarray] = []
    if evidence_mode == "software_verification":
        frames = _collect_synthetic(resolution, int(cfg.get("synthetic_frames", 8)))
        input_source = "synthetic"
    else:
        if images_dir is None or not images_dir.is_dir():
            print("real_evidence requires images_dir with images (no synthetic fallback)", file=sys.stderr)
            return 2
        frames = _collect_from_dir(images_dir, resolution)
        if not frames:
            print(f"no images in {images_dir}", file=sys.stderr)
            return 2
        input_source = "images_dir"

    process = (
        process_frame_historical
        if profile in ("historical_exact", "software_verification_historical")
        else process_frame_current_production
    )
    pipeline_id = PIPELINE_ID if process is process_frame_historical else "current_production_enhancement_fusion"
    source_commit = SOURCE_COMMIT if process is process_frame_historical else git_head()

    for i in range(warmup):
        process(frames[i % len(frames)], target_res=resolution)

    raw_rows: List[Dict[str, Any]] = []
    totals: List[float] = []
    cores: List[float] = []
    stage_sums: Dict[str, List[float]] = {}

    for i in range(timed_n):
        t_decode0 = time.perf_counter_ns()
        rgb = frames[i % len(frames)]
        t_decode1 = time.perf_counter_ns()
        combined, dets, stages = process(rgb, target_res=resolution)
        stages = dict(stages)
        stages["image_decode"] = (t_decode1 - t_decode0) / 1e9
        stages["total_pipeline"] = stages.get("total_pipeline", 0.0) + stages["image_decode"]
        totals.append(float(stages["total_pipeline"]))
        cores.append(float(stages["core_processing"]))
        for k, v in stages.items():
            stage_sums.setdefault(k, []).append(float(v))
        row = {"iter": i, "n_detections": len(dets), "map_mean": float(np.mean(combined))}
        row.update({f"{k}_s": float(v) for k, v in stages.items()})
        raw_rows.append(row)

    mean_total = float(statistics.fmean(totals))
    mean_core = float(statistics.fmean(cores))
    if profile in ("historical_exact", "software_verification_historical"):
        headline_name = "historical_exact_fps"
        headline_fps = 1.0 / mean_total
        historical_fps = headline_fps
        current_fps = None
    else:
        headline_name = "current_pipeline_fps"
        headline_fps = 1.0 / mean_total
        historical_fps = None
        current_fps = headline_fps

    sw = evidence_mode == "software_verification"
    summary: Dict[str, Any] = {
        "claim_ids": cfg.get("claim_ids", ["C07"]),
        "evidence_class": "software_verification" if sw else "candidate_real_evidence",
        "eligible_for_claim_closure": False,
        "input_source": input_source,
        "pipeline_id": pipeline_id,
        "source_commit": source_commit,
        "implementation_hash": implementation_hash(),
        "equivalence_test_status": "see_tests",
        "input_manifest_sha256": None,
        "config_sha256": sha256_file(config_path),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "input_resolution": resolution,
        "batch_size": batch_size,
        "learned_depth_enabled": False,
        "autoencoder_enabled": False,
        "warmup_count": warmup,
        "timed_iteration_count": timed_n,
        "frame_count": len(frames),
        "mean_core_latency_s": mean_core,
        "mean_total_latency_s": mean_total,
        "headline_fps": headline_fps,
        "headline_metric_name": headline_name,
        "historical_exact_fps": historical_fps,
        "current_pipeline_fps": current_fps,
        "stages": {k: float(statistics.fmean(v)) for k, v in stage_sums.items()},
        "environment_extended": _extended_env(),
        "profile": profile,
        "notes": (
            "Does not claim equality with accepted-abstract 28.1 FPS. "
            f"Headline={headline_name}."
        ),
    }

    run_id = args.run_id or make_run_id("c07")
    out_dir = resolve_repo_path(str(cfg["output_directory"])) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in raw_rows for k in r.keys()})
    with (out_dir / "timing_raw.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)
    write_json(out_dir / "timing_summary.json", summary)
    write_run_bundle(
        out_dir,
        config_path=config_path,
        command=sys.argv,
        provenance_extra={
            "accepted_abstract_28_1_not_used_as_pass_fail": True,
            "evidence_class": summary["evidence_class"],
            "eligible_for_claim_closure": False,
        },
    )
    print("=== C07 benchmark ===")
    print(f"profile           : {profile}")
    print(f"pipeline_id       : {pipeline_id}")
    print(f"input_source      : {input_source}")
    print(f"headline          : {headline_name}={headline_fps:.2f}")
    print(f"evidence_class    : {summary['evidence_class']}")
    print(f"artifacts         : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
