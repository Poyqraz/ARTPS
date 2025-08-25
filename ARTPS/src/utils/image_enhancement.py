from typing import Dict, Tuple, List
import numpy as np
import cv2
from PIL import Image


def _to_rgb_uint8(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _estimate_noise_sigma(rgb_u8: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    high_freq = gray.astype(np.float32) - blur.astype(np.float32)
    return float(np.std(high_freq))


def _estimate_sharpness(rgb_u8: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(lap_var)


def _estimate_contrast(rgb_u8: np.ndarray) -> float:
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    return float(np.std(L))


def estimate_quality_metrics(rgb_u8: np.ndarray) -> Dict[str, float]:
    return {
        "noise_sigma": _estimate_noise_sigma(rgb_u8),
        "sharpness": _estimate_sharpness(rgb_u8),
        "contrast": _estimate_contrast(rgb_u8),
        "mean_brightness": float(np.mean(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY))),
    }


def _auto_white_balance_grayworld(rgb_u8: np.ndarray) -> np.ndarray:
    # Gray-world white balance
    mean_b = np.mean(rgb_u8[..., 2])
    mean_g = np.mean(rgb_u8[..., 1])
    mean_r = np.mean(rgb_u8[..., 0])
    mean_gray = (mean_r + mean_g + mean_b) / 3.0 + 1e-6
    scale_r = mean_gray / (mean_r + 1e-6)
    scale_g = mean_gray / (mean_g + 1e-6)
    scale_b = mean_gray / (mean_b + 1e-6)
    r = np.clip(rgb_u8[..., 0] * scale_r, 0, 255)
    g = np.clip(rgb_u8[..., 1] * scale_g, 0, 255)
    b = np.clip(rgb_u8[..., 2] * scale_b, 0, 255)
    out = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return out


def _apply_clahe_lab(rgb_u8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid_size, tile_grid_size))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def _unsharp_mask(rgb_u8: np.ndarray, strength: float = 1.0, radius: int = 3) -> np.ndarray:
    blur = cv2.GaussianBlur(rgb_u8, (0, 0), sigmaX=radius, sigmaY=radius)
    sharp = cv2.addWeighted(rgb_u8, 1 + strength, blur, -strength, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _auto_gamma(rgb_u8: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    gray_mean = np.mean(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)) + 1e-6
    gamma = np.clip(np.log(target_mean / 255.0 + 1e-6) / np.log(gray_mean / 255.0 + 1e-6), 0.5, 2.0)
    x = rgb_u8.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def _denoise(rgb_u8: np.ndarray, h: int = 5) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(rgb_u8, None, h, h, 7, 21)


def _upscale(rgb_u8: np.ndarray, target_long_side: int = 1024) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    long_side = max(h, w)
    if long_side >= target_long_side:
        return rgb_u8
    scale = target_long_side / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    up = cv2.resize(rgb_u8, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # Hafif detay artırma
    try:
        up = cv2.detailEnhance(up, sigma_s=10, sigma_r=0.15)
    except Exception:
        pass
    return up


def enhance_image_auto(pil_image: Image.Image,
                       config: Dict = None) -> Tuple[Image.Image, Dict[str, float], Dict[str, float], List[str]]:
    """Görüntüyü otomatik olarak iyileştirir ve rapor döner.

    Dönenler: enhanced_pil, metrics_before, metrics_after, steps
    """
    if config is None:
        config = {}

    params = {
        "enable_upscale": config.get("enable_upscale", True),
        "target_long_side": int(config.get("target_long_side", 1024)),
        "enable_denoise": config.get("enable_denoise", True),
        "denoise_h": int(config.get("denoise_h", 5)),
        "enable_awb": config.get("enable_awb", True),
        "enable_clahe": config.get("enable_clahe", True),
        "clahe_clip": float(config.get("clahe_clip", 2.0)),
        "enable_gamma": config.get("enable_gamma", True),
        "target_mean": float(config.get("target_mean", 128.0)),
        "enable_sharpen": config.get("enable_sharpen", True),
        "sharpen_strength": float(config.get("sharpen_strength", 0.6)),
        "sharpen_radius": int(config.get("sharpen_radius", 2)),
    }

    rgb = _to_rgb_uint8(pil_image)
    steps: List[str] = []
    before = estimate_quality_metrics(rgb)

    # 1) Upscale (önce yapılır, böylece sonraki işlemler daha iyi çalışır)
    if params["enable_upscale"]:
        rgb = _upscale(rgb, params["target_long_side"])
        steps.append(f"upscale({params['target_long_side']})")

    # 2) Denoise (gürültü yüksekse uygula)
    if params["enable_denoise"]:
        noise_sigma = _estimate_noise_sigma(rgb)
        if noise_sigma > 6.0:
            rgb = _denoise(rgb, params["denoise_h"])
            steps.append(f"denoise(h={params['denoise_h']})")

    # 3) AWB (renk dengesi)
    if params["enable_awb"]:
        rgb = _auto_white_balance_grayworld(rgb)
        steps.append("awb(grayworld)")

    # 4) Kontrast (CLAHE)
    if params["enable_clahe"]:
        rgb = _apply_clahe_lab(rgb, params["clahe_clip"], 8)
        steps.append(f"clahe(clip={params['clahe_clip']})")

    # 5) Gamma (pozlama)
    if params["enable_gamma"]:
        rgb = _auto_gamma(rgb, params["target_mean"])
        steps.append(f"gamma(target_mean={params['target_mean']})")

    # 6) Keskinlik (Unsharp)
    if params["enable_sharpen"]:
        rgb = _unsharp_mask(rgb, params["sharpen_strength"], params["sharpen_radius"])
        steps.append(f"unsharp(s={params['sharpen_strength']},r={params['sharpen_radius']})")

    after = estimate_quality_metrics(rgb)
    enhanced_pil = Image.fromarray(rgb)
    return enhanced_pil, before, after, steps


