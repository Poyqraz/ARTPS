import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import torch


class _Spinner:
    def __init__(self, *_args, **_kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


class _StStub:
    def __init__(self):
        self.session_state: Dict[str, object] = {}
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def success(self, *args, **kwargs):
        pass
    def spinner(self, *args, **kwargs):
        return _Spinner()
    # Placeholders used by app; not needed here but kept to avoid attribute errors
    def markdown(self, *args, **kwargs):
        pass
    def image(self, *args, **kwargs):
        pass
    def set_page_config(self, *args, **kwargs):
        pass
    def cache_resource(self, func=None, **_kwargs):
        # Basit dekoratör: fonksiyonu aynen döndür (cache yok)
        if func is None:
            def decorator(f):
                return f
            return decorator
        return func


def _overlay(base_rgb: np.ndarray, heat_norm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = base_rgb.shape[:2]
    heat_col = plt.cm.inferno(heat_norm)[..., :3]
    base = base_rgb.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    if heat_col.shape[:2] != (h, w):
        heat_col = np.array(Image.fromarray((heat_col * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    out = (1.0 - alpha) * base + alpha * heat_col
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def _det_to_jsonable(det: dict) -> dict:
    keep = {
        "x", "y", "w", "h", "score", "score_raw", "score_policy",
        "object_anomaly_score", "object_value_score", "combined_pool",
        "padim_pool", "patchcore_pool", "depth_pool", "detector_conf",
        "class_id", "class_name", "proposal_source", "recommended",
        "cluster_id", "sim_max", "score_drop", "in_priority_buffer",
        "comb_mean", "edge_mean", "z_peak", "z_mean", "z_std", "depth_span",
    }
    row = {k: det[k] for k in keep if k in det}
    poly = det.get("poly")
    if poly is not None:
        row["poly"] = poly
    return row


def _draw_detection_boxes(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    disp = image.copy()
    for i, det in enumerate(detections, start=1):
        x, y, w, h = det['x'], det['y'], det['w'], det['h']
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"#{i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(disp, (x, max(0, y - th - 6)), (x + tw + 6, y - 2), (0, 255, 0), -1)
        cv2.putText(disp, label, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return disp


def _class_from_path(pth: Path) -> str:
    name = pth.name
    if "__" in name:
        return name.split("__", 1)[0]
    parts = pth.parts
    for i, part in enumerate(parts):
        if part == "valid" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def main():
    p = argparse.ArgumentParser(description="App analiz akışıyla doğru kutulama overlay'leri üret")
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='results/paper_figs/detection_overlays_app')
    p.add_argument('--backend', type=str, default='heuristic', choices=['heuristic', 'yolo', 'hybrid'])
    p.add_argument('--device', type=str, default='auto', choices=['cuda', 'cpu', 'auto'])
    p.add_argument('--detector-conf', type=float, default=0.25)
    p.add_argument('--jsonl', type=str, default='results/app_detections.jsonl')
    p.add_argument('--recall-ablation', type=str, default='slim',
                   choices=['full', 'slim', 'no_boost', 'no_cap'])
    args = p.parse_args()

    device_pref = args.device
    if device_pref == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA istendi ancak torch.cuda.is_available() False döndü")

    # streamlit stub enjekte et
    sys.modules['streamlit'] = _StStub()

    # Proje kökünü sys.path'e ekle
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import app as appmod  # type: ignore

    # Modelleri yükle
    load_result = appmod.load_models(device_preference=None if device_pref == "auto" else device_pref)
    models = load_result["models"]
    if models is None:
        raise RuntimeError("Modeller yüklenemedi; results/*.pth kontrol edin.")
    jsonl_path = Path(args.jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink()

    root = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in root.rglob('*') if p.suffix.lower() in exts]
    latencies: list[float] = []

    for pth in files:
        t0 = time.perf_counter()
        img = Image.open(pth).convert('RGB')
        appmod.__dict__.update({
            "detector_backend": args.backend,
            "detector_conf": float(args.detector_conf),
            "recall_ablation": args.recall_ablation,
        })
        res = appmod.analyze_mars_image(models, img)
        base_u8 = (res['original'] * 255).astype(np.uint8)
        heat = res['combined_anomaly_map'].astype(np.float32)
        overlay = _overlay(base_u8, heat, alpha=0.45)
        detections = res.get('detections') or []
        disp = _draw_detection_boxes(overlay, detections)
        Image.fromarray(disp).save(out_dir / f"{pth.stem}_det_overlay_app.png")
        depth_overlay_path = None
        depth_overlay = res.get("depth_rgb_overlay")
        if isinstance(depth_overlay, np.ndarray):
            depth_overlay_path = out_dir / f"{pth.stem}_depth_overlay_app.png"
            Image.fromarray(_draw_detection_boxes(depth_overlay.astype(np.uint8), detections)).save(depth_overlay_path)
        depth_edge_overlay_path = None
        depth_map_full = res.get("depth_map_full")
        depth_edge_overlay = appmod._compute_depth_edge_overlay(base_u8, depth_map_full, alpha=0.45)
        if isinstance(depth_edge_overlay, np.ndarray):
            depth_edge_overlay_path = out_dir / f"{pth.stem}_depth_edge_overlay_app.png"
            Image.fromarray(_draw_detection_boxes(depth_edge_overlay.astype(np.uint8), detections)).save(depth_edge_overlay_path)
        protrusion_overlay_path = None
        protrusion_map = res.get("depth_protrusion_map")
        if isinstance(protrusion_map, np.ndarray):
            protrusion_rgb = appmod._blend_heat_on_rgb(base_u8, protrusion_map.astype(np.float32), alpha=0.50, cmap="inferno")
            protrusion_overlay_path = out_dir / f"{pth.stem}_protrusion_overlay_app.png"
            Image.fromarray(_draw_detection_boxes(protrusion_rgb, detections)).save(protrusion_overlay_path)
        rows: List[dict] = []
        for det in detections:
            rows.append(_det_to_jsonable(det))
        record = {
            "image_path": str(pth),
            "class_label": _class_from_path(pth),
            "backend": str(res.get("detector_backend", args.backend)),
            "proposal_count": int(res.get("proposal_count", len(rows))),
            "pre_filter_proposal_count": int(res.get("pre_filter_proposal_count", len(rows))),
            "clutter_mode": bool(res.get("clutter_mode", False)),
            "rocky_recall_mode": bool(res.get("rocky_recall_mode", False)),
            "proposal_sources_breakdown": res.get("proposal_sources_breakdown") or {},
            "latency_sec": float(time.perf_counter() - t0),
            "anomaly_mse": float(res.get("anomaly_score", 0.0) or 0.0),
            "combined_anomaly_score": float(res.get("combined_anomaly_score", 0.0) or 0.0),
            "known_value_score": float(res.get("known_value_score", 0.5) or 0.5),
            "detections": rows,
            "recommended_targets": res.get("recommended_targets") or [],
            "priority_buffer": res.get("priority_buffer") or [],
            "depth_overlay_path": str(depth_overlay_path) if depth_overlay_path else None,
            "depth_edge_overlay_path": str(depth_edge_overlay_path) if depth_edge_overlay_path else None,
            "protrusion_overlay_path": str(protrusion_overlay_path) if protrusion_overlay_path else None,
            "viz_quality": res.get("viz_quality") or {},
        }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        latencies.append(float(record["latency_sec"]))
    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    print(f"Overlay üretildi: {out_dir} (adet: {len(files)}, avg_latency_sec={avg_lat:.2f})")


if __name__ == '__main__':
    main()


