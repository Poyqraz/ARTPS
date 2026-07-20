from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

# Weight path (gitignored via raw_models/ and *.pth)
_REALESRGAN_WEIGHT = Path(__file__).resolve().parents[2] / "raw_models" / "RealESRGAN_x2plus.pth"
# ponytail: lazy singleton; False = load attempted and failed
_realesrgan_upsampler: Any = None


@dataclass
class EnhancementResult:
    image: Image.Image
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    steps: List[str] = field(default_factory=list)
    realesrgan_used: bool = False
    realesrgan_fallback: bool = False  # requested but cubic used instead


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
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _estimate_contrast(rgb_u8: np.ndarray) -> float:
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    return float(np.std(lab[..., 0]))


def estimate_quality_metrics(rgb_u8: np.ndarray) -> Dict[str, float]:
    return {
        "noise_sigma": _estimate_noise_sigma(rgb_u8),
        "sharpness": _estimate_sharpness(rgb_u8),
        "contrast": _estimate_contrast(rgb_u8),
        "mean_brightness": float(np.mean(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY))),
    }


def _auto_white_balance_grayworld(rgb_u8: np.ndarray) -> np.ndarray:
    mean_b = np.mean(rgb_u8[..., 2])
    mean_g = np.mean(rgb_u8[..., 1])
    mean_r = np.mean(rgb_u8[..., 0])
    mean_gray = (mean_r + mean_g + mean_b) / 3.0 + 1e-6
    r = np.clip(rgb_u8[..., 0] * (mean_gray / (mean_r + 1e-6)), 0, 255)
    g = np.clip(rgb_u8[..., 1] * (mean_gray / (mean_g + 1e-6)), 0, 255)
    b = np.clip(rgb_u8[..., 2] * (mean_gray / (mean_b + 1e-6)), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _apply_clahe_lab(rgb_u8: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid_size, tile_grid_size))
    return cv2.cvtColor(cv2.merge([clahe.apply(L), A, B]), cv2.COLOR_LAB2RGB)


def _unsharp_mask(rgb_u8: np.ndarray, strength: float = 1.0, radius: int = 3) -> np.ndarray:
    blur = cv2.GaussianBlur(rgb_u8, (0, 0), sigmaX=radius, sigmaY=radius)
    return np.clip(cv2.addWeighted(rgb_u8, 1 + strength, blur, -strength, 0), 0, 255).astype(np.uint8)


def _auto_gamma(rgb_u8: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    gray_mean = np.mean(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)) + 1e-6
    gamma = np.clip(np.log(target_mean / 255.0 + 1e-6) / np.log(gray_mean / 255.0 + 1e-6), 0.5, 2.0)
    return np.clip((rgb_u8.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)


def _denoise(rgb_u8: np.ndarray, h: int = 5) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(rgb_u8, None, h, h, 7, 21)


def _upscale(rgb_u8: np.ndarray, target_long_side: int = 1024, detail_enhance: bool = True) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    long_side = max(h, w)
    if long_side >= target_long_side:
        return rgb_u8
    scale = target_long_side / float(long_side)
    up = cv2.resize(
        rgb_u8,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    if detail_enhance:
        try:
            up = cv2.detailEnhance(up, sigma_s=10, sigma_r=0.15)
        except Exception:
            pass
    return up


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_realesrgan_upsampler(tile: int = 400):
    """Lazy-load RealESRGANer once per process; None if unavailable."""
    global _realesrgan_upsampler
    if _realesrgan_upsampler is False:
        return None
    if _realesrgan_upsampler is not None:
        return _realesrgan_upsampler
    if not _REALESRGAN_WEIGHT.is_file():
        _realesrgan_upsampler = False
        return None
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        from realesrgan import RealESRGANer  # type: ignore

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        _realesrgan_upsampler = RealESRGANer(
            scale=2,
            model_path=str(_REALESRGAN_WEIGHT),
            model=model,
            tile=int(tile),
            tile_pad=10,
            pre_pad=0,
            half=_cuda_available(),
        )
        return _realesrgan_upsampler
    except Exception:
        _realesrgan_upsampler = False
        return None


def _upscale_realesrgan(rgb_u8: np.ndarray, *, outscale: int = 2, tile: int = 400) -> Optional[np.ndarray]:
    """Optional RealESRGAN_x2plus; returns None if pkg/weight missing or inference fails."""
    upsampler = _get_realesrgan_upsampler(tile=tile)
    if upsampler is None:
        return None
    try:
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        out_bgr, _ = upsampler.enhance(bgr, outscale=int(outscale))
        if out_bgr is None or out_bgr.size == 0:
            return None
        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        # OOM / runtime: one retry with smaller tiles, then give up
        if int(tile) > 200:
            global _realesrgan_upsampler
            _realesrgan_upsampler = None
            return _upscale_realesrgan(rgb_u8, outscale=outscale, tile=200)
        return None


# Mars yüzeyi + hibrit anomali analizi için güvenli varsayılanlar.
# Gray-world AWB Mars kırmızı tonunu nötralize eder → eğitim dağılımından sapma / sahte anomali riski.
ENHANCE_PROFILES: Dict[str, Dict] = {
    "generic": {
        "enable_upscale": True,
        "target_long_side": 1024,
        "enable_denoise": True,
        "denoise_h": 5,
        "enable_awb": True,
        "enable_clahe": True,
        "clahe_clip": 2.0,
        "enable_gamma": True,
        "target_mean": 128.0,
        "enable_sharpen": True,
        "sharpen_strength": 0.6,
        "sharpen_radius": 2,
        "detail_enhance": True,
        "enable_realesrgan": False,
    },
    "mars": {
        "enable_upscale": True,
        "target_long_side": 1024,
        "enable_denoise": True,
        "denoise_h": 5,
        "enable_awb": False,
        "enable_clahe": True,
        "clahe_clip": 1.5,
        "enable_gamma": True,
        "target_mean": 115.0,
        "enable_sharpen": True,
        "sharpen_strength": 0.35,
        "sharpen_radius": 2,
        "detail_enhance": False,
        "enable_realesrgan": False,
    },
}


def enhance_image_auto(
    pil_image: Image.Image,
    config: Dict | None = None,
    profile: str = "mars",
) -> EnhancementResult:
    """Görüntüyü otomatik iyileştirir; varsayılan Mars hibrit-analiz güvenli profilidir."""
    p = dict(ENHANCE_PROFILES.get(profile, ENHANCE_PROFILES["mars"]))
    if config:
        p.update(config)

    enable_upscale = bool(p["enable_upscale"])
    target_long_side = int(p["target_long_side"])
    enable_denoise = bool(p["enable_denoise"])
    denoise_h = int(p["denoise_h"])
    enable_awb = bool(p["enable_awb"])
    enable_clahe = bool(p["enable_clahe"])
    clahe_clip = float(p["clahe_clip"])
    enable_gamma = bool(p["enable_gamma"])
    target_mean = float(p["target_mean"])
    enable_sharpen = bool(p["enable_sharpen"])
    sharpen_strength = float(p["sharpen_strength"])
    sharpen_radius = int(p["sharpen_radius"])
    detail_enhance = bool(p["detail_enhance"])
    enable_realesrgan = bool(p["enable_realesrgan"])

    rgb = _to_rgb_uint8(pil_image)
    steps: List[str] = [f"profile({profile})"]
    before = estimate_quality_metrics(rgb)
    realesrgan_used = False
    realesrgan_fallback = False

    # 1) Denoise first so SR does not amplify noise in deep shadows
    if enable_denoise and _estimate_noise_sigma(rgb) > 6.0:
        rgb = _denoise(rgb, denoise_h)
        steps.append(f"denoise(h={denoise_h})")

    # 2) Upscale: optional SR x2, then CUBIC to target_long_side if still short
    if enable_upscale and max(rgb.shape[:2]) < target_long_side:
        if enable_realesrgan:
            sr = _upscale_realesrgan(rgb, outscale=2, tile=400)
            if sr is not None:
                rgb = sr
                steps.append("realesrgan(x2plus,outscale=2)")
                realesrgan_used = True
            else:
                realesrgan_fallback = True
        if max(rgb.shape[:2]) < target_long_side:
            rgb = _upscale(rgb, target_long_side, detail_enhance=detail_enhance)
            steps.append(f"upscale({target_long_side},detail={detail_enhance})")

    if enable_awb:
        rgb = _auto_white_balance_grayworld(rgb)
        steps.append("awb(grayworld)")

    if enable_clahe:
        rgb = _apply_clahe_lab(rgb, clahe_clip, 8)
        steps.append(f"clahe(clip={clahe_clip})")

    if enable_gamma:
        rgb = _auto_gamma(rgb, target_mean)
        steps.append(f"gamma(target_mean={target_mean})")

    if enable_sharpen:
        rgb = _unsharp_mask(rgb, sharpen_strength, sharpen_radius)
        steps.append(f"unsharp(s={sharpen_strength},r={sharpen_radius})")

    return EnhancementResult(
        image=Image.fromarray(rgb),
        metrics_before=before,
        metrics_after=estimate_quality_metrics(rgb),
        steps=steps,
        realesrgan_used=realesrgan_used,
        realesrgan_fallback=realesrgan_fallback,
    )
