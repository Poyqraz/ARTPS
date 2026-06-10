import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import torch

try:
    from src.models.optimized_autoencoder import OptimizedAutoencoder
    from src.models.depth_enhanced_classifier import DepthEnhancedClassifier
    from src.models.depth_estimation import MiDaSDepthEstimator
    from src.models.anomaly import PaDiM, PaDiMConfig, PatchCore, PatchCoreConfig
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.models.optimized_autoencoder import OptimizedAutoencoder
    from src.models.depth_enhanced_classifier import DepthEnhancedClassifier
    from src.models.depth_estimation import MiDaSDepthEstimator
    from src.models.anomaly import PaDiM, PaDiMConfig, PatchCore, PatchCoreConfig


def _normalize_map(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


@torch.no_grad()
def _ae_forward(ae: OptimizedAutoencoder, image: Image.Image, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    img_rz = image.resize((128, 128), Image.LANCZOS)
    arr = np.array(img_rz, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).to(device)
    reconstructed, latent = ae(x)
    rec = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mse = float(np.mean((arr - rec) ** 2))
    return mse, arr, rec, latent.squeeze().cpu().numpy()


def _overlay_heatmap(base_rgb: np.ndarray, heat_norm: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    h, w = base_rgb.shape[:2]
    heat_col = plt.cm.inferno(heat_norm)[..., :3]
    base = base_rgb.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    if heat_col.shape[:2] != (h, w):
        heat_col = np.array(Image.fromarray((heat_col * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    out = (1.0 - alpha) * base + alpha * heat_col
    return np.clip(out, 0.0, 1.0)


def _nms(boxes: List[Tuple[int, int, int, int, float]], iou_thresh: float) -> List[Tuple[int, int, int, int, float]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep: List[Tuple[int, int, int, int, float]] = []
    def iou(a, b):
        ax1, ay1, aw, ah, _ = a
        bx1, by1, bw, bh, _ = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return float(inter / max(1e-6, union))
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if iou(best, b) < iou_thresh]
    return keep


def _compute_combined_map(orig: np.ndarray, recon: np.ndarray, depth: np.ndarray,
                          w_recon: float = 0.50, w_depth: float = 0.30, w_texture: float = 0.20,
                          w_lap: float = 0.08, w_detail: float = 0.12,
                          alpha_shad: float = 0.65, beta_illum: float = 0.25,
                          padim_map: np.ndarray = None, patchcore_map: np.ndarray = None,
                          w_padim: float = 0.30, w_patchcore: float = 0.25) -> np.ndarray:
    H, W = depth.shape[:2]
    o = cv2.resize(orig.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    r = cv2.resize(recon.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    recon_diff = ((o - r) ** 2).mean(axis=2)
    recon_n = _normalize_map(recon_diff)
    img_u8 = (o * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    grad_n = _normalize_map(grad_mag)
    shadow_n = _normalize_map(1.0 - gray)
    d = depth.astype(np.float32)
    dx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    depth_edge = np.sqrt(dx ** 2 + dy ** 2)
    depth_n = _normalize_map(depth_edge)
    depth_lap = _normalize_map(np.abs(cv2.Laplacian(d, cv2.CV_32F, ksize=3)))
    # detail
    lap3 = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap5 = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
    dog = cv2.GaussianBlur(gray, (0, 0), 0.8) - cv2.GaussianBlur(gray, (0, 0), 1.6)
    detail = _normalize_map(0.33 * np.abs(lap3) + 0.33 * np.abs(lap5) + 0.34 * np.abs(dog))
    # combine
    texture = 0.35 * shadow_n + 0.65 * grad_n
    combined = (
        w_recon * recon_n +
        w_depth * depth_n +
        w_texture * texture +
        w_lap * depth_lap +
        w_detail * detail
    )
    combined = np.clip(combined, 0.0, 1.0)
    # fuse external maps
    if padim_map is not None:
        combined = np.clip((1.0 - w_padim) * combined + w_padim * padim_map, 0.0, 1.0)
    if patchcore_map is not None:
        combined = np.clip((1.0 - w_patchcore) * combined + w_patchcore * patchcore_map, 0.0, 1.0)
    return combined


def _detect_from_map(heat: np.ndarray, hyst_high_pct: int = 97, hyst_low_pct: int = 92,
                     min_area_ratio: float = 0.001, nms_iou: float = 0.35, top_k: int = 25) -> List[Tuple[int, int, int, int, float]]:
    H, W = heat.shape[:2]
    hi = float(np.percentile(heat, hyst_high_pct))
    lo = float(np.percentile(heat, hyst_low_pct))
    strong = (heat >= hi).astype(np.uint8)
    weak = ((heat >= lo) & (heat < hi)).astype(np.uint8)
    mask = (strong * 255)
    # connect weak with strong via a simple dilation pass
    kernel = np.ones((3, 3), np.uint8)
    weak_dil = cv2.dilate(strong, kernel, iterations=1)
    mask = np.where((weak == 1) & (weak_dil == 1), 255, mask).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = min_area_ratio * H * W
    boxes: List[Tuple[int, int, int, int, float]] = []
    for c in contours:
        if cv2.contourArea(c) < area_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        score = float(heat[y:y+h, x:x+w].mean())
        boxes.append((x, y, w, h, score))
    boxes = _nms(boxes, nms_iou)
    boxes = boxes[:top_k]
    return boxes


def _load_models(device: torch.device) -> Dict[str, object]:
    models: Dict[str, object] = {}
    # AE
    ae = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    try:
        ckpt = torch.load('results/optimized_autoencoder_curiosity_extended.pth', map_location=device, weights_only=True)  # type: ignore
    except TypeError:
        ckpt = torch.load('results/optimized_autoencoder_curiosity_extended.pth', map_location=device)
    ae.load_state_dict(ckpt['model_state_dict'])
    ae.to(device).eval()
    models['ae'] = ae
    # Depth
    models['depth'] = MiDaSDepthEstimator(model_type='DPT_Large', device=device)
    # Optional anomaly fusion models
    try:
        padim_stats = Path('results/padim_stats.pth')
        if padim_stats.exists():
            pad = PaDiM(PaDiMConfig(image_size=256))
            pad.load(str(padim_stats))
            models['padim'] = pad
    except Exception:
        pass
    try:
        pcore_bank = Path('results/patchcore_bank.pth')
        if pcore_bank.exists():
            pc = PatchCore(PatchCoreConfig(image_size=256))
            pc.load(str(pcore_bank))
            models['patchcore'] = pc
    except Exception:
        pass
    return models


def process_image(pth: Path, models: Dict[str, object], device: torch.device, out_dir: Path) -> None:
    img = Image.open(pth).convert('RGB')
    mse, arr128, rec128, _ = _ae_forward(models['ae'], img, device)
    arr_full = np.array(img, dtype=np.float32) / 255.0
    depth_map, _ = models['depth'].estimate_depth(arr_full, apply_enhancement=True, high_detail=True, tta_flips=True, use_fgs=True, use_wmf=True)
    # external maps
    padim_map = None
    patchcore_map = None
    try:
        base_u8 = (np.array(img).astype(np.uint8))
        if 'padim' in models:
            padim_map = models['padim'].predict_anomaly_map(base_u8)
        if 'patchcore' in models:
            patchcore_map = models['patchcore'].predict_anomaly_map(base_u8)
    except Exception:
        pass
    comb = _compute_combined_map(arr128, rec128, depth_map, padim_map=padim_map, patchcore_map=patchcore_map)
    # detections
    boxes = _detect_from_map(comb, hyst_high_pct=97, hyst_low_pct=92, min_area_ratio=0.001, nms_iou=0.35, top_k=25)
    # overlay
    overlay = (255 * _overlay_heatmap(np.array(img), comb, alpha=0.45)).astype(np.uint8)
    disp = overlay.copy()
    for i, (x, y, w, h, score) in enumerate(boxes, start=1):
        color = (0, 255, 0)
        cv2.rectangle(disp, (x, y), (x + w, y + h), color, 2)
        label = f"#{i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(disp, (x, max(0, y - th - 6)), (x + tw + 6, y - 2), color, -1)
        cv2.putText(disp, label, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(disp).save(out_dir / f"{pth.stem}_det_overlay.png")
    # also save raw combined heatmap
    plt.figure(figsize=(5, 4))
    plt.imshow(comb, cmap='inferno')
    plt.title('Combined Anomaly')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"{pth.stem}_combined.png", dpi=140)
    plt.close()


def main():
    p = argparse.ArgumentParser(description='Anomali tespiti için kutulanmış overlay üretici')
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='results/paper_figs/detection_overlays')
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else ('cuda' if args.device == 'cuda' else 'cpu'))
    models = _load_models(device)

    root = Path(args.images_dir)
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in root.rglob('*') if p.suffix.lower() in exts]
    out_dir = Path(args.out_dir)
    count = 0
    for pth in files:
        process_image(pth, models, device, out_dir)
        count += 1
    print(f"Overlay üretildi: {out_dir} (adet: {count})")


if __name__ == '__main__':
    main()


