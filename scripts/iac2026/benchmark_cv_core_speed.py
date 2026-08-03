#!/usr/bin/env python3
"""C07 workstation speed harness — historical_exact vs enhancement surrogate profiles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    environment_snapshot,
    git_dirty,
    git_head,
    load_json_schema,
    make_run_id,
    read_csv_dicts,
    resolve_repo_path,
    resolve_under_dataset_root,
    sha256_file,
    write_json,
    write_run_bundle,
)
from _config import ConfigValidationError, load_timing_config, require_jsonschema  # noqa: E402
from cv_core_pipeline import (  # noqa: E402
    CURRENT_SURROGATE_PIPELINE_ID,
    PIPELINE_ID,
    SOURCE_COMMIT,
    implementation_hash,
    process_frame_current_enhancement_historical_surrogate,
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
    # Varied sizes so timed resize inside process_frame is non-trivial
    frames = []
    for i in range(max(1, n)):
        side = resolution if i % 2 == 0 else resolution + 64
        frames.append(rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8))
    return frames


def _load_manifest_frames(
    *,
    manifest_path: Path,
    dataset_root: Path,
) -> Tuple[List[np.ndarray], str, str, int]:
    rows = read_csv_dicts(manifest_path)
    required = {
        "sample_id",
        "relative_path",
        "sha256",
        "source_id",
        "mission",
        "instrument",
        "sol",
        "order_index",
    }
    if not rows:
        raise ValueError("input manifest is empty")
    missing_cols = required - set(rows[0].keys())
    if missing_cols:
        raise ValueError(f"input manifest missing columns: {sorted(missing_cols)}")

    sample_ids = [r["sample_id"] for r in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample_id in input manifest")
    order_indices = [int(r["order_index"]) for r in rows]
    if len(order_indices) != len(set(order_indices)):
        raise ValueError("duplicate order_index in input manifest")

    ordered = sorted(rows, key=lambda r: int(r["order_index"]))
    frames: List[np.ndarray] = []
    path_digests: List[str] = []
    for r in ordered:
        path = resolve_under_dataset_root(dataset_root, r["relative_path"])
        if not path.is_file():
            raise ValueError(f"missing input file: {path}")
        digest = sha256_file(path)
        if digest.lower() != str(r["sha256"]).lower():
            raise ValueError(f"sha256 mismatch for {path.name}")
        # Preload original resolution — resize happens inside timed process_frame
        img = Image.open(path).convert("RGB")
        frames.append(np.asarray(img, dtype=np.uint8))
        path_digests.append(f"{r['sample_id']}:{digest}:{r['order_index']}")

    manifest_sha = sha256_file(manifest_path)
    ordered_set_sha = hashlib.sha256("\n".join(path_digests).encode("utf-8")).hexdigest()
    return frames, manifest_sha, ordered_set_sha, len(frames)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C07 core speed harness")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--software-verification", action="store_true")
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
    allow_dirty = bool(cfg.get("allow_dirty_git", False))

    if evidence_mode == "real_evidence" and args.software_verification:
        print("real_evidence forbids --software-verification", file=sys.stderr)
        return 2
    if evidence_mode == "software_verification" and not args.software_verification:
        print("software_verification requires --software-verification", file=sys.stderr)
        return 2
    if git_dirty() and not allow_dirty:
        print("dirty git rejected", file=sys.stderr)
        return 2

    PROFILE_MAP = {
        "historical_exact": process_frame_historical,
        "historical_software_verification": process_frame_historical,
        "current_enhancement_historical_surrogate": process_frame_current_enhancement_historical_surrogate,
    }
    if profile not in PROFILE_MAP:
        print(f"unknown profile {profile!r}", file=sys.stderr)
        return 2
    process = PROFILE_MAP[profile]

    input_manifest_sha: Optional[str] = None
    ordered_input_set_sha: Optional[str] = None
    input_file_count = 0

    if evidence_mode == "software_verification":
        frames = _collect_synthetic(resolution, int(cfg.get("synthetic_frames", 8)))
        input_source = "synthetic"
        input_file_count = len(frames)
    else:
        if cfg.get("images_dir") and not cfg.get("input_manifest"):
            print("real_evidence rejects images_dir-only; provide input_manifest", file=sys.stderr)
            return 2
        manifest_path = resolve_repo_path(str(cfg["input_manifest"]))
        root_env = str(cfg["dataset_root_env"])
        root_val = os.environ.get(root_env)
        if not root_val:
            print(f"dataset_root_env {root_env!r} is not set", file=sys.stderr)
            return 2
        try:
            frames, input_manifest_sha, ordered_input_set_sha, input_file_count = _load_manifest_frames(
                manifest_path=manifest_path,
                dataset_root=Path(root_val),
            )
        except (ValueError, OSError) as exc:
            print(f"manifest load failed: {exc}", file=sys.stderr)
            return 2
        input_source = "manifest"

    pipeline_id = (
        PIPELINE_ID if process is process_frame_historical else CURRENT_SURROGATE_PIPELINE_ID
    )
    source_commit = SOURCE_COMMIT if process is process_frame_historical else git_head()

    for i in range(warmup):
        process(frames[i % len(frames)], target_res=resolution)

    raw_rows: List[Dict[str, Any]] = []
    totals: List[float] = []
    cores: List[float] = []
    stage_sums: Dict[str, List[float]] = {}

    for i in range(timed_n):
        t_fetch0 = time.perf_counter_ns()
        rgb = frames[i % len(frames)]
        t_fetch1 = time.perf_counter_ns()
        combined, dets, stages = process(rgb, target_res=resolution)
        stages = dict(stages)
        stages["frame_fetch"] = (t_fetch1 - t_fetch0) / 1e9
        # Headline total_pipeline is process_frame scope (excludes frame_fetch / disk decode)
        totals.append(float(stages["total_pipeline"]))
        cores.append(float(stages["core_processing"]))
        for k, v in stages.items():
            stage_sums.setdefault(k, []).append(float(v))
        row = {"iter": i, "n_detections": len(dets), "map_mean": float(np.mean(combined))}
        row.update({f"{k}_s": float(v) for k, v in stages.items()})
        raw_rows.append(row)

    mean_total = float(statistics.fmean(totals))
    mean_core = float(statistics.fmean(cores))
    if profile in ("historical_exact", "historical_software_verification"):
        headline_name = "historical_exact_fps"
        headline_fps = 1.0 / mean_total
        historical_fps = headline_fps
        current_fps = None
    else:
        headline_name = "current_enhancement_historical_surrogate_fps"
        headline_fps = 1.0 / mean_total
        historical_fps = None
        current_fps = headline_fps

    sw = evidence_mode == "software_verification"
    cfg_sha = sha256_file(config_path)
    impl_hash = implementation_hash()
    summary: Dict[str, Any] = {
        "claim_ids": cfg.get("claim_ids", ["C07"]),
        "evidence_class": "software_verification" if sw else "candidate_real_evidence",
        "eligible_for_claim_closure": False,
        "input_source": input_source,
        "pipeline_id": pipeline_id,
        "profile": profile,
        "source_commit": source_commit,
        "implementation_hash": impl_hash,
        "equivalence_test_status": "not_independently_verified",
        "input_manifest_sha256": input_manifest_sha if not sw else None,
        "input_file_count": input_file_count,
        "ordered_input_set_sha256": ordered_input_set_sha if not sw else None,
        "config_sha256": cfg_sha,
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
        "notes": (
            "Does not claim equality with accepted-abstract 28.1 FPS. "
            f"Headline={headline_name}. "
            "Disk enumeration/decode excluded from headline; resize is inside timed process_frame. "
            "Stage fusion_localization_combined is measured as one block (no fabricated 70/30)."
        ),
    }

    require_jsonschema()
    import jsonschema

    # Soften SW timing schema: input_manifest_sha256 may be null
    try:
        jsonschema.Draft202012Validator(load_json_schema("timing_result.schema.json")).validate(summary)
    except jsonschema.ValidationError as exc:
        print(f"timing summary schema validation failed: {exc.message}", file=sys.stderr)
        return 2

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
