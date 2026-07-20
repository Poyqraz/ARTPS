"""Small/far detection focus tiles: adaptive crop + 4-panel quality."""
from __future__ import annotations

import numpy as np

import app as app_mod
from app import (
    _auto_enhance_focus,
    _fit_to_guide,
    _focus_crop_bounds,
    _precompute_focus_tiles,
)


def test_focus_crop_bounds_expands_tiny_bbox() -> None:
    x1, y1, x2, y2, _, _ = _focus_crop_bounds(100, 100, 12, 12, 256, 256)
    assert (x2 - x1) >= 64
    assert (y2 - y1) >= 64


def test_fit_to_guide_resizes() -> None:
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    out = _fit_to_guide(img, (40, 60))
    assert out.shape[:2] == (40, 60)


def test_auto_enhance_small_crop_upscales_without_sr(monkeypatch) -> None:
    monkeypatch.setattr(app_mod, "_upscale_realesrgan", lambda *a, **k: None)
    tiny = np.random.RandomState(0).randint(0, 255, (24, 24, 3), dtype=np.uint8)
    out = _auto_enhance_focus(
        tiny,
        scale=max(2.0, 96 / 24),
        interp_code=1,
        amount=0.4,
        try_realesrgan=True,
    )
    assert max(out.shape[:2]) >= 48


def test_precompute_focus_tiles_four_panel_aspect(monkeypatch) -> None:
    monkeypatch.setattr(app_mod, "_upscale_realesrgan", lambda *a, **k: None)
    H = W = 128
    combined = np.full((H, W), 0.2, np.float32)
    combined[40:52, 40:52] = 0.9
    original = np.clip(
        np.random.RandomState(1).rand(H, W, 3).astype(np.float32) * 0.5 + 0.25,
        0,
        1,
    )
    depth = np.linspace(0.1, 0.9, H, dtype=np.float32)[:, None].repeat(W, axis=1)
    results = {
        "combined_anomaly_map": combined,
        "original": original,
        "depth_map_full": depth,
        "depth_rgb_overlay": None,
    }
    dets = [{"x": 44, "y": 44, "w": 8, "h": 8}]
    app_mod.focus_h = 120
    app_mod.focus_overlay = True
    app_mod.focus_sharpen = True
    app_mod.focus_interp = "INTER_LINEAR"

    tiles = _precompute_focus_tiles(results, dets)
    assert len(tiles) == 1
    tile = tiles[0]
    assert tile is not None and tile.ndim == 3
    aspect = float(tile.shape[1]) / float(tile.shape[0])
    assert 3.2 <= aspect <= 4.8
    assert tile.shape[0] == 120


def test_large_bbox_skips_forced_sr_path(monkeypatch) -> None:
    calls = {"sr": 0}

    def fake_sr(rgb, outscale=2, tile=200):
        calls["sr"] += 1
        return None

    monkeypatch.setattr(app_mod, "_upscale_realesrgan", fake_sr)
    big = np.zeros((120, 120, 3), dtype=np.uint8)
    _auto_enhance_focus(big, scale=1.0, interp_code=1, amount=0.4, try_realesrgan=True)
    assert calls["sr"] == 0


def test_heat_depth_fit_match_guide() -> None:
    guide = np.zeros((64, 80, 3), dtype=np.uint8)
    heat = np.zeros((20, 25, 3), dtype=np.uint8)
    fitted = _fit_to_guide(heat, guide.shape[:2])
    assert fitted.shape[:2] == guide.shape[:2]
