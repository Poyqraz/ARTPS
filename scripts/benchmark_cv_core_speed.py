import argparse
import os
import platform
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image


def _normalize_map(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if float(hi - lo) < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _auto_gamma(rgb_u8: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    gray_mean = float(np.mean(gray)) + 1e-6
    gamma = float(np.clip(np.log(target_mean / 255.0 + 1e-6) / np.log(gray_mean / 255.0 + 1e-6), 0.5, 2.0))
    x = rgb_u8.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def _enhance_rgb_u8(rgb_u8: np.ndarray) -> np.ndarray:
    """
    Lightweight enhancement approximating the paper's preprocessing:
    bilateral denoise + CLAHE (LAB) + gamma + unsharp mask.
    """
    # Edge-preserving denoise (bilateral)
    den = cv2.bilateralFilter(rgb_u8, d=7, sigmaColor=35, sigmaSpace=35)

    # CLAHE on L channel (LAB)
    lab = cv2.cvtColor(den, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    rgb = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)

    # Auto gamma
    rgb = _auto_gamma(rgb, target_mean=128.0)

    # Unsharp mask
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    sharp = cv2.addWeighted(rgb, 1.6, blur, -0.6, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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


def _nms(dets: List[dict], iou_thr: float = 0.35, top_k: int = 25) -> List[dict]:
    dets = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    keep: List[dict] = []
    for d in dets:
        if len(keep) >= int(top_k):
            break
        box_d = (int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]))
        if any(_iou_xywh(box_d, (int(k["x"]), int(k["y"]), int(k["w"]), int(k["h"]))) > iou_thr for k in keep):
            continue
        keep.append(d)
    return keep


def _compute_combined_map_and_detections(
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
    # Assumption: inputs already share the same spatial size.
    H, W = depth_map_f.shape[:2]
    orig = original_rgb_f.astype(np.float32)
    recon = reconstructed_rgb_f.astype(np.float32)

    # Reconstruction diff
    recon_diff = ((orig - recon) ** 2).mean(axis=2)
    recon_diff_n = _normalize_map(recon_diff)

    # Image cues
    img_u8 = np.clip(orig * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_n = _normalize_map(np.sqrt(sobelx * sobelx + sobely * sobely))
    shadow_n = _normalize_map(1.0 - gray)

    # Depth cues
    depth = depth_map_f.astype(np.float32)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_edge_n = _normalize_map(np.sqrt(dx * dx + dy * dy))
    depth_lap_n = _normalize_map(np.abs(cv2.Laplacian(depth, cv2.CV_32F, ksize=3)))

    # Fine detail
    lap3 = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap5 = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
    dog = cv2.GaussianBlur(gray, (0, 0), 0.8) - cv2.GaussianBlur(gray, (0, 0), 1.6)
    fine_detail = _normalize_map(np.abs(lap3) + 0.6 * np.abs(lap5) + 0.8 * np.abs(dog))

    texture_term = 0.35 * shadow_n + 0.65 * grad_mag_n
    combined = (
        w_recon * recon_diff_n
        + w_depth * depth_edge_n
        + w_texture * texture_term
        + w_lap * depth_lap_n
        + w_detail * fine_detail
    )
    combined = np.clip(combined, 0.0, 1.0)

    # Hysteresis-like thresholding
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

    dets = _nms(dets, iou_thr=float(nms_iou), top_k=int(top_k))
    return combined, dets


def _fallback_depth_from_gray(gray_f: np.ndarray) -> np.ndarray:
    sx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sx * sx + sy * sy)
    return _normalize_map(1.0 - grad)


def process_frame(pil_img: Image.Image, *, target_res: int = 768) -> None:
    # Resize normalization (paper default path uses 768 for depth input in app)
    img = pil_img.convert("RGB").resize((target_res, target_res), Image.LANCZOS)

    # Enhancement (OpenCV stage; Torch-free)
    orig_u8 = _enhance_rgb_u8(np.array(img, dtype=np.uint8))
    orig_f = orig_u8.astype(np.float32) / 255.0

    # Lightweight "reconstruction surrogate" (no Torch): blur as a proxy
    recon_f = cv2.GaussianBlur(orig_f, ksize=(0, 0), sigmaX=1.2, sigmaY=1.2)

    gray_f = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    depth_f = _fallback_depth_from_gray(gray_f)

    _compute_combined_map_and_detections(orig_f, recon_f, depth_f)


def _collect_images(img_dir: Path, limit: int) -> List[Image.Image]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    paths = sorted(paths)[: max(1, int(limit))]
    imgs: List[Image.Image] = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark OpenCV-only core pipeline speed (no Torch).")
    ap.add_argument("--images_dir", type=str, default=str(Path("results") / "paper_images"))
    ap.add_argument("--target_res", type=int, default=768)
    ap.add_argument("--rounds", type=int, default=10, help="Timed rounds over the image set.")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    img_dir = Path(args.images_dir)
    if not img_dir.exists():
        raise SystemExit(f"images_dir not found: {img_dir}")

    imgs = _collect_images(img_dir, limit=int(args.limit))
    if not imgs:
        raise SystemExit(f"No images found in: {img_dir}")

    # Warmup
    for _ in range(int(args.warmup)):
        process_frame(imgs[0], target_res=int(args.target_res))

    # Timed
    t0 = time.perf_counter()
    frames = 0
    for _ in range(int(args.rounds)):
        for im in imgs:
            process_frame(im, target_res=int(args.target_res))
            frames += 1
    dt = time.perf_counter() - t0
    fps = frames / max(1e-9, dt)
    ms = (dt / max(1, frames)) * 1000.0

    # Environment metadata
    proc_id = os.environ.get("PROCESSOR_IDENTIFIER", "") or platform.processor()
    print("=== OpenCV-core benchmark (no Torch) ===")
    print(f"frames            : {frames}")
    print(f"resolution        : {args.target_res}x{args.target_res}")
    print(f"total_time_sec    : {dt:.4f}")
    print(f"avg_latency_ms    : {ms:.2f}")
    print(f"fps               : {fps:.2f}")
    print("--- environment ---")
    print(f"python            : {platform.python_version()}")
    print(f"os               : {platform.platform()}")
    print(f"cpu               : {proc_id}")
    print(f"numpy             : {np.__version__}")
    print(f"opencv            : {cv2.__version__}")


if __name__ == "__main__":
    main()

