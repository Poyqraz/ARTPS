from __future__ import annotations

import json
import numpy as np
import subprocess
import sys
from pathlib import Path

from scripts.export_app_detections import _det_to_jsonable


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_make_proxy_labels_detection_mode(tmp_path: Path) -> None:
    inp = tmp_path / "detections.jsonl"
    out = tmp_path / "proxy.jsonl"
    _write_jsonl(
        inp,
        [
            {
                "image_path": "img_a.png",
                "backend": "heuristic",
                "anomaly_mse": 0.01,
                "roughness": 0.2,
                "depth_variance": 0.1,
                "known_value_score": 0.4,
                "detections": [
                    {
                        "x": 10,
                        "y": 12,
                        "w": 20,
                        "h": 18,
                        "score": 0.8,
                        "object_anomaly_score": 0.75,
                        "object_value_score": 0.55,
                        "detector_conf": 0.82,
                        "proposal_source": "heuristic",
                    }
                ],
            }
        ],
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_proxy_labels.py"),
            "--in",
            str(inp),
            "--out",
            str(out),
            "--mode",
            "detections",
        ],
        check=True,
        cwd=ROOT,
    )
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["image_path"] == "img_a.png"
    assert row["bbox"] == [10, 12, 20, 18]
    assert 0.0 <= row["proxy_label"] <= 1.0


def test_run_benchmarks_summarizes_recall_and_precision(tmp_path: Path) -> None:
    preds = tmp_path / "preds.jsonl"
    gt = tmp_path / "gt.jsonl"
    out_dir = tmp_path / "bench"
    _write_jsonl(
        preds,
        [
            {
                "image_path": "img_a.png",
                "backend": "heuristic",
                "detections": [{"x": 10, "y": 10, "w": 20, "h": 20}],
            },
            {
                "image_path": "img_b.png",
                "backend": "yolo",
                "detections": [{"x": 40, "y": 40, "w": 10, "h": 10}],
            },
        ],
    )
    _write_jsonl(
        gt,
        [
            {"image_path": "img_a.png", "boxes": [[10, 10, 20, 20]]},
            {"image_path": "img_b.png", "boxes": [[42, 42, 8, 8]]},
        ],
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_benchmarks.py"),
            "--out",
            str(out_dir),
            "--predictions",
            str(preds),
            "--ground-truth",
            str(gt),
        ],
        check=True,
        cwd=ROOT,
    )
    summary = json.loads((out_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["proposal_recall"] == 1.0
    assert summary["bbox_precision"] == 1.0
    assert set(summary["backend_breakdown"].keys()) == {"heuristic", "yolo"}


def test_export_app_detections_keeps_geomorph_metrics() -> None:
    row = _det_to_jsonable(
        {
            "x": 1,
            "y": 2,
            "w": 3,
            "h": 4,
            "score": 0.9,
            "z_peak": 0.8,
            "z_mean": 0.4,
            "z_std": 0.1,
            "depth_span": 0.6,
            "unused_field": "ignore-me",
        }
    )
    assert row["z_peak"] == 0.8
    assert row["z_mean"] == 0.4
    assert row["z_std"] == 0.1
    assert row["depth_span"] == 0.6


def test_compute_depth_edge_overlay_produces_uint8_image() -> None:
    sys.path.insert(0, str(ROOT))
    import app as appmod  # type: ignore

    # Deterministik sentetik depth: merkezde daha yüksek değer -> kenar gücü değişir.
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.zeros((32, 32), dtype=np.float32)
    depth[8:24, 8:24] = 1.0
    overlay = appmod._compute_depth_edge_overlay(base, depth, alpha=0.5)
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == base.shape
    assert overlay.dtype == np.uint8


def test_ensure_depth_viz_assets_backfills_before_qc() -> None:
    sys.path.insert(0, str(ROOT))
    import app as appmod  # type: ignore

    h, w = 48, 48
    depth = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
    depth[10:30, 10:30] += 0.4
    base = np.full((h, w, 3), 70, dtype=np.uint8)
    results = {
        "original": base.astype(np.float32) / 255.0,
        "depth_map_full": depth,
        "detections": [{"x": 2, "y": 2, "w": 8, "h": 8, "depth_span": 0.4}],
        "focus_tiles": [np.zeros((24, 96, 3), dtype=np.uint8)],
        "viz_quality": {
            "status": "fail",
            "score": 0.0,
            "checks": {"depth_rgb_overlay": False},
            "metrics": {},
            "messages": ["stale"],
        },
    }
    # Stale fail + missing overlays → ensure + re-eval path used by UI helper
    had = isinstance(results.get("depth_rgb_overlay"), np.ndarray)
    appmod._ensure_depth_viz_assets(results)
    assert isinstance(results["depth_rgb_overlay"], np.ndarray)
    assert isinstance(results["depth_protrusion_map"], np.ndarray)
    if not had:
        results["viz_quality"] = appmod._evaluate_depth_viz_quality(results)
    assert results["viz_quality"]["checks"]["depth_rgb_overlay"] is True
    assert results["viz_quality"]["checks"]["geomorph_metrics"] is True


def test_evaluate_depth_viz_quality_pass_and_fail() -> None:
    sys.path.insert(0, str(ROOT))
    import app as appmod  # type: ignore

    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w]
    depth = (xx.astype(np.float32) / w) + 0.4 * ((yy - h / 2) ** 2 < 100).astype(np.float32)
    base = np.full((h, w, 3), 80, dtype=np.uint8)
    overlay = appmod._compute_depth_rgb_overlay(base, depth)
    protr = appmod._compute_protrusion_map(depth)
    tile = np.zeros((40, 160, 3), dtype=np.uint8)  # ~4:1 aspect → 4-panel
    dets = [
        {
            "x": 8,
            "y": 8,
            "w": 16,
            "h": 16,
            "z_peak": 0.7,
            "z_mean": 0.3,
            "z_std": 0.1,
            "depth_span": 0.5,
        }
    ]
    ok = appmod._evaluate_depth_viz_quality(
        {
            "depth_rgb_overlay": overlay,
            "depth_protrusion_map": protr,
            "depth_map_full": depth,
            "focus_tiles": [tile],
            "detections": dets,
        }
    )
    assert ok["status"] == "pass"
    assert ok["score"] >= 0.85
    assert ok["checks"]["depth_rgb_overlay"] is True

    bad = appmod._evaluate_depth_viz_quality(
        {
            "depth_rgb_overlay": None,
            "depth_protrusion_map": None,
            "depth_map_full": None,
            "focus_tiles": [],
            "detections": [{"x": 1, "y": 1, "w": 2, "h": 2}],
        }
    )
    assert bad["status"] == "fail"
    assert bad["checks"]["depth_rgb_overlay"] is False
