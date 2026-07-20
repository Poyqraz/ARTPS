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


def test_realesrgan_runs_after_denoise_before_clahe(monkeypatch) -> None:
    """Denoise → SR → CLAHE sırası (yüksek gürültü + ESRGAN mock)."""
    import src.utils.image_enhancement as ie

    order: list[str] = []

    def fake_denoise(rgb, h=5):
        order.append("denoise")
        return rgb

    def fake_sr(rgb, outscale=2, tile=400):
        order.append("realesrgan")
        h, w = rgb.shape[:2]
        return np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)[: h * 2, : w * 2]

    monkeypatch.setattr(ie, "_denoise", fake_denoise)
    monkeypatch.setattr(ie, "_upscale_realesrgan", fake_sr)
    monkeypatch.setattr(ie, "_estimate_noise_sigma", lambda rgb: 20.0)

    noisy = np.random.RandomState(1).randint(0, 255, (40, 40, 3), dtype=np.uint8)
    result = ie.enhance_image_auto(
        Image.fromarray(noisy),
        {
            "enable_realesrgan": True,
            "enable_upscale": True,
            "enable_denoise": True,
            "enable_clahe": True,
            "target_long_side": 160,
        },
        profile="mars",
    )
    assert "denoise" in order
    assert "realesrgan" in order
    assert order.index("denoise") < order.index("realesrgan")
    steps = result.steps
    den_i = next(i for i, s in enumerate(steps) if s.startswith("denoise"))
    sr_i = next(i for i, s in enumerate(steps) if s.startswith("realesrgan"))
    cla_i = next(i for i, s in enumerate(steps) if s.startswith("clahe"))
    assert den_i < sr_i < cla_i


def test_realesrgan_before_clahe_when_no_denoise(monkeypatch) -> None:
    import src.utils.image_enhancement as ie

    def fake_sr(rgb, outscale=2, tile=400):
        h, w = rgb.shape[:2]
        return np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    monkeypatch.setattr(ie, "_upscale_realesrgan", fake_sr)
    monkeypatch.setattr(ie, "_estimate_noise_sigma", lambda rgb: 1.0)

    result = ie.enhance_image_auto(
        _mars_red_image(40),
        {
            "enable_realesrgan": True,
            "enable_upscale": True,
            "enable_denoise": True,
            "enable_clahe": True,
            "target_long_side": 160,
        },
        profile="mars",
    )
    steps = result.steps
    assert not any(s.startswith("denoise") for s in steps)
    sr_i = next(i for i, s in enumerate(steps) if s.startswith("realesrgan"))
    cla_i = next(i for i, s in enumerate(steps) if s.startswith("clahe"))
    assert sr_i < cla_i
