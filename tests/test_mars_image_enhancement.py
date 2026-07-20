"""Mars-safe otomatik iyileştirme: hibrit analizi bozmamalı."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.utils.image_enhancement import ENHANCE_PROFILES, enhance_image_auto


def _mars_red_image(size: int = 64) -> Image.Image:
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 0] = 170
    rgb[..., 1] = 85
    rgb[..., 2] = 45
    noise = np.random.RandomState(0).randint(0, 20, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def test_mars_profile_preserves_red_cast_and_skips_awb() -> None:
    result = enhance_image_auto(_mars_red_image(), profile="mars")
    step_text = " ".join(result.steps)
    assert "profile(mars)" in step_text
    assert "awb(" not in step_text
    assert ENHANCE_PROFILES["mars"]["enable_awb"] is False
    arr = np.asarray(result.image)
    assert float(arr[..., 0].mean()) > float(arr[..., 1].mean())
    assert float(arr[..., 0].mean()) > float(arr[..., 2].mean())


def test_generic_profile_can_run_awb() -> None:
    result = enhance_image_auto(_mars_red_image(), profile="generic")
    assert any(s.startswith("awb(") for s in result.steps)


def test_mars_realesrgan_default_off() -> None:
    assert ENHANCE_PROFILES["mars"].get("enable_realesrgan", False) is False
    result = enhance_image_auto(_mars_red_image(48), profile="mars")
    assert result.realesrgan_used is False
    assert "realesrgan(" not in " ".join(result.steps)


def test_realesrgan_request_via_public_api() -> None:
    """enable_realesrgan=True: used veya fallback; CUBIC target_long_side'a tamamlar."""
    result = enhance_image_auto(
        _mars_red_image(48),
        {"enable_realesrgan": True, "enable_upscale": True, "target_long_side": 96},
        profile="mars",
    )
    arr = np.asarray(result.image)
    assert max(arr.shape[:2]) >= 90
    if result.realesrgan_used:
        assert result.realesrgan_fallback is False
        assert any(s.startswith("realesrgan(") for s in result.steps)
    else:
        assert result.realesrgan_fallback is True
