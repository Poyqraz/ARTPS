"""Historical OpenCV surrogate pipeline recovered from commit 8f7e3ff.

pipeline_id: historical_opencv_surrogate_8f7e3ff

This is NOT the current production core path. Do not label outputs as
matching accepted-abstract 28.1 FPS without a closed measured run.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

PIPELINE_ID = "historical_opencv_surrogate_8f7e3ff"
SOURCE_COMMIT = "8f7e3ff"


def normalize_map(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if float(hi - lo) < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def auto_gamma(rgb_u8: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    gray_mean = float(np.mean(gray)) + 1e-6
    gamma = float(
        np.clip(
            np.log(target_mean / 255.0 + 1e-6) / np.log(gray_mean / 255.0 + 1e-6),
            0.5,
            2.0,
        )
    )
    x = rgb_u8.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def enhance_rgb_u8(rgb_u8: np.ndarray) -> np.ndarray:
    den = cv2.bilateralFilter(rgb_u8, d=7, sigmaColor=35, sigmaSpace=35)
    lab = cv2.cvtColor(den, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    rgb = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)
    rgb = auto_gamma(rgb, target_mean=128.0)
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    sharp = cv2.addWeighted(rgb, 1.6, blur, -0.6, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return float(inter / max(1e-6, union))


def nms(dets: List[dict], iou_thr: float = 0.35, top_k: int = 25) -> List[dict]:
    dets = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    keep: List[dict] = []
    for d in dets:
        if len(keep) >= int(top_k):
            break
        box_d = (int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]))
        if any(
            iou_xywh(box_d, (int(k["x"]), int(k["y"]), int(k["w"]), int(k["h"]))) > iou_thr
            for k in keep
        ):
            continue
        keep.append(d)
    return keep


def fallback_depth_from_gray(gray_f: np.ndarray) -> np.ndarray:
    sx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sx * sx + sy * sy)
    return normalize_map(1.0 - grad)


def compute_combined_map_and_detections(
    original_rgb_f: np.ndarray,
    reconstructed_rgb_f: np.ndarray,
    depth_map_f: np.ndarray,
    *,
    hyst_high_pct: int = 97,
    hyst_low_pct: int = 92,
    nms_iou: float = 0.35,
    top_k: int = 25,
    w_recon: float = 0.50,
    w_depth: float = 0.30,
    w_texture: float = 0.20,
    w_lap: float = 0.08,
    w_detail: float = 0.12,
) -> Tuple[np.ndarray, List[dict]]:
    H, W = depth_map_f.shape[:2]
    orig = original_rgb_f.astype(np.float32)
    recon = reconstructed_rgb_f.astype(np.float32)

    recon_diff = ((orig - recon) ** 2).mean(axis=2)
    recon_diff_n = normalize_map(recon_diff)

    img_u8 = np.clip(orig * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_n = normalize_map(np.sqrt(sobelx * sobelx + sobely * sobely))
    shadow_n = normalize_map(1.0 - gray)

    depth = depth_map_f.astype(np.float32)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_edge_n = normalize_map(np.sqrt(dx * dx + dy * dy))
    depth_lap_n = normalize_map(np.abs(cv2.Laplacian(depth, cv2.CV_32F, ksize=3)))

    lap3 = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap5 = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
    dog = cv2.GaussianBlur(gray, (0, 0), 0.8) - cv2.GaussianBlur(gray, (0, 0), 1.6)
    fine_detail = normalize_map(np.abs(lap3) + 0.6 * np.abs(lap5) + 0.8 * np.abs(dog))

    texture_term = 0.35 * shadow_n + 0.65 * grad_mag_n
    combined = np.clip(
        w_recon * recon_diff_n
        + w_depth * depth_edge_n
        + w_texture * texture_term
        + w_lap * depth_lap_n
        + w_detail * fine_detail,
        0.0,
        1.0,
    )

    high_th = float(np.percentile(combined, hyst_high_pct))
    low_th = float(np.percentile(combined, hyst_low_pct))
    high_mask = (combined >= high_th).astype(np.uint8) * 255
    low_mask = (combined >= low_th).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    seeds = high_mask.copy()
    prev = np.zeros_like(seeds)
    for _ in range(10):
        dil = cv2.dilate(seeds, kernel, iterations=1)
        seeds = cv2.bitwise_and(dil, low_mask)
        if np.array_equal(seeds, prev):
            break
        prev = seeds.copy()

    mask = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: List[dict] = []
    area_min = 0.001 * H * W
    for c in contours:
        if cv2.contourArea(c) < area_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        y2 = min(H, y + h)
        x2 = min(W, x + w)
        score = float(np.mean(combined[y:y2, x:x2])) if (y2 > y and x2 > x) else 0.0
        dets.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": score})

    dets = nms(dets, iou_thr=float(nms_iou), top_k=int(top_k))
    return combined, dets


def process_frame_historical(
    rgb_u8: np.ndarray, *, target_res: int = 256
) -> Tuple[np.ndarray, List[dict], Dict[str, float]]:
    """Historical process_frame scope with per-stage timings (seconds).

    Timed scope includes resize + enhance + recon surrogate + fallback depth +
    fusion_localization_combined. Disk decode is outside this function.
    """
    import time

    stages: Dict[str, float] = {}
    t0 = time.perf_counter_ns()
    if rgb_u8.shape[0] != target_res or rgb_u8.shape[1] != target_res:
        resized = cv2.resize(rgb_u8, (target_res, target_res), interpolation=cv2.INTER_AREA)
    else:
        resized = rgb_u8
    t1 = time.perf_counter_ns()
    stages["resize_preprocess"] = (t1 - t0) / 1e9

    t0 = time.perf_counter_ns()
    enhanced = enhance_rgb_u8(resized)
    t1 = time.perf_counter_ns()
    stages["enhancement"] = (t1 - t0) / 1e9

    orig_f = enhanced.astype(np.float32) / 255.0
    t0 = time.perf_counter_ns()
    recon_f = cv2.GaussianBlur(orig_f, ksize=(0, 0), sigmaX=1.2, sigmaY=1.2)
    t1 = time.perf_counter_ns()
    stages["reconstruction_surrogate"] = (t1 - t0) / 1e9

    gray_f = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    t0 = time.perf_counter_ns()
    depth_f = fallback_depth_from_gray(gray_f)
    t1 = time.perf_counter_ns()
    stages["fallback_depth"] = (t1 - t0) / 1e9

    t0 = time.perf_counter_ns()
    combined, dets = compute_combined_map_and_detections(orig_f, recon_f, depth_f)
    t1 = time.perf_counter_ns()
    stages["fusion_localization_combined"] = (t1 - t0) / 1e9
    stages["core_processing"] = (
        stages["enhancement"]
        + stages["reconstruction_surrogate"]
        + stages["fallback_depth"]
        + stages["fusion_localization_combined"]
    )
    stages["total_pipeline"] = stages["resize_preprocess"] + stages["core_processing"]
    return combined, dets, stages


def core_process_rgb_u8(rgb_u8: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    combined, dets, _ = process_frame_historical(rgb_u8, target_res=rgb_u8.shape[0])
    return combined, dets


def implementation_hash() -> str:
    path = Path(__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


CURRENT_SURROGATE_PIPELINE_ID = "current_enhancement_historical_surrogate"


def process_frame_current_enhancement_historical_surrogate(
    rgb_u8: np.ndarray, *, target_res: int = 256
) -> Tuple[np.ndarray, List[dict], Dict[str, float]]:
    """Supplementary profile: current enhancement + historical recon/depth/fusion.

    Not the accepted C07 claim path. Not a full current-production pipeline.
    """
    import sys
    import time

    from PIL import Image

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.utils.image_enhancement import enhance_image_auto

    stages: Dict[str, float] = {}
    t0 = time.perf_counter_ns()
    if rgb_u8.shape[0] != target_res or rgb_u8.shape[1] != target_res:
        resized = cv2.resize(rgb_u8, (target_res, target_res), interpolation=cv2.INTER_AREA)
    else:
        resized = rgb_u8
    t1 = time.perf_counter_ns()
    stages["resize_preprocess"] = (t1 - t0) / 1e9

    t0 = time.perf_counter_ns()
    result = enhance_image_auto(
        Image.fromarray(resized),
        config={
            "enable_realesrgan": False,
            "enable_upscale": False,
            "enable_denoise": True,
            "enable_clahe": True,
            "enable_gamma": True,
            "enable_sharpen": True,
        },
        profile="mars",
    )
    enhanced = np.asarray(result.image.convert("RGB"), dtype=np.uint8)
    if enhanced.shape[0] != target_res:
        enhanced = cv2.resize(enhanced, (target_res, target_res), interpolation=cv2.INTER_AREA)
    t1 = time.perf_counter_ns()
    stages["enhancement"] = (t1 - t0) / 1e9

    orig_f = enhanced.astype(np.float32) / 255.0
    t0 = time.perf_counter_ns()
    recon_f = cv2.GaussianBlur(orig_f, ksize=(0, 0), sigmaX=1.2, sigmaY=1.2)
    t1 = time.perf_counter_ns()
    stages["reconstruction_surrogate"] = (t1 - t0) / 1e9

    gray_f = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    t0 = time.perf_counter_ns()
    depth_f = fallback_depth_from_gray(gray_f)
    t1 = time.perf_counter_ns()
    stages["fallback_depth"] = (t1 - t0) / 1e9

    t0 = time.perf_counter_ns()
    combined, dets = compute_combined_map_and_detections(orig_f, recon_f, depth_f)
    t1 = time.perf_counter_ns()
    stages["fusion_localization_combined"] = (t1 - t0) / 1e9
    stages["core_processing"] = (
        stages["enhancement"]
        + stages["reconstruction_surrogate"]
        + stages["fallback_depth"]
        + stages["fusion_localization_combined"]
    )
    stages["total_pipeline"] = stages["resize_preprocess"] + stages["core_processing"]
    return combined, dets, stages


# Back-compat alias (deprecated name)
process_frame_current_production = process_frame_current_enhancement_historical_surrogate
