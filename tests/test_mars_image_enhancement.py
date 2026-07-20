"""Mars-safe otomatik iyileştirme: hibrit analizi bozmamalı."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.utils.image_enhancement import (
    ENHANCE_PROFILES,
    _upscale_realesrgan,
    enhance_image_auto,
)


def _mars_red_image(size: int = 64) -> Image.Image:
    # Tipik Mars kırmızı-turuncu yüzey baskınlığı
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = 170
    rgb[..., 1] = 85
    rgb[..., 2] = 45
    # hafif gürültü / düşük kontrastlı yapı
    noise = np.random.RandomState(0).randint(0, 20, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def test_mars_profile_preserves_red_cast_and_skips_awb() -> None:
    img = _mars_red_image()
    enhanced, _before, _after, steps = enhance_image_auto(img, profile="mars")
    step_text = " ".join(steps)
    assert "profile(mars)" in step_text
    assert "awb(" not in step_text
    assert ENHANCE_PROFILES["mars"]["enable_awb"] is False

    arr = np.asarray(enhanced)
    assert float(arr[..., 0].mean()) > float(arr[..., 1].mean())
    assert float(arr[..., 0].mean()) > float(arr[..., 2].mean())


def test_generic_profile_can_run_awb() -> None:
    img = _mars_red_image()
    _enhanced, _before, _after, steps = enhance_image_auto(img, profile="generic")
    assert any(s.startswith("awb(") for s in steps)


def test_mars_realesrgan_default_off() -> None:
    assert ENHANCE_PROFILES["mars"].get("enable_realesrgan", False) is False
    img = _mars_red_image(48)
    _enhanced, _before, _after, steps = enhance_image_auto(img, profile="mars")
    assert "realesrgan(" not in " ".join(steps)


def test_upscale_realesrgan_skip_or_doubles() -> None:
    """Weight/pkg yoksa None; varsa çıktı en az ~2× boyut."""
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 160
    rgb[..., 1] = 80
    rgb[..., 2] = 40
    out = _upscale_realesrgan(rgb, outscale=2, tile=64)
    if out is None:
        return  # paket veya raw_models/RealESRGAN_x2plus.pth yok — beklenen
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[0] >= 60 and out.shape[1] >= 60
