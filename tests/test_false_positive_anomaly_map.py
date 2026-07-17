import numpy as np

from app import compute_combined_anomaly_map


def test_boundary_connected_rover_shadow_does_not_survive_as_large_detection():
    rgb = np.full((96, 96, 3), 0.62, np.float32)
    depth = np.full((96, 96), 0.5, np.float32)

    rgb[:, 78:] = 0.10
    depth[:, 78:] = 0.15
    rgb[62:, :28] = 0.06
    reconstructed = rgb.copy()

    _, detections = compute_combined_anomaly_map(
        rgb,
        reconstructed,
        depth,
        hyst_high_pct=90,
        hyst_low_pct=80,
        top_k=20,
        edge_reinforce=0.8,
    )

    assert all(not (det["x"] >= 74 and det["w"] > 8) for det in detections)
    assert all(not (det["y"] >= 58 and det["x"] < 30) for det in detections)


def test_border_textured_rover_part_suppressed_center_relatively_preserved():
    """Sinira bagli dokulu/yakin rover parcasi (tekerlek/kol) bastirilir; ayni
    sahnedeki merkez (sinira bagli olmayan) yuksek-recon bolge goreli korunur.

    Not: Bu pipeline kucuk sentetik sahnelerde lokalize tespit uretmedigi icin
    (mevcut testler de yalnizca 'yokluk' olcer) hedefli bastirma harita
    duzeyinde dogrulanir.
    """
    rng = np.random.default_rng(1)
    rgb = (0.55 + rng.normal(0.0, 0.01, size=(96, 96, 3))).astype(np.float32)
    depth = (0.5 + rng.normal(0.0, 0.02, size=(96, 96))).astype(np.float32)

    # Saga yaslanik, dokulu ve yakin rover parcasi: guclu depth sapmasi + recon hatasi
    depth[:, 80:] = 0.92
    rover_texture = rng.uniform(0.0, 1.0, size=(96, 16)).astype(np.float32)
    for c in range(3):
        rgb[:, 80:, c] = rover_texture

    reconstructed = rgb.copy()
    reconstructed[:, 80:] = 0.5
    # Merkezde dokulu, recon hatali ve derinlik cikintili anomali (sinira bagli degil)
    center_texture = rng.uniform(0.2, 0.9, size=(16, 16)).astype(np.float32)
    for c in range(3):
        rgb[40:56, 40:56, c] = center_texture
    reconstructed[40:56, 40:56] = 0.55
    depth[40:56, 40:56] = 0.62

    combined, detections = compute_combined_anomaly_map(
        rgb,
        reconstructed,
        depth,
        hyst_high_pct=90,
        hyst_low_pct=80,
        top_k=25,
        edge_reinforce=0.6,
    )

    # Rover bolgesinde hicbir tespit hayatta kalmaz
    assert all(not (det["x"] + det["w"] > 78) for det in detections)
    # Bastirma hedeflidir: rover bolgesi merkeze gore belirgin sekilde bastirilir
    center_mean = float(combined[40:56, 40:56].mean())
    rover_mean = float(combined[:, 80:].mean())
    assert center_mean > 1.5 * rover_mean


def test_distant_mid_scene_target_survives_better_than_horizon_band():
    """Uzak ama sahne ortasindaki hedef, ufuk bandindaki zayif yapilardan goreli
    olarak daha guclu kalmali."""
    rng = np.random.default_rng(7)
    rgb = np.full((96, 96, 3), 0.58, np.float32)
    rgb += rng.normal(0.0, 0.01, size=rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)

    depth = np.full((96, 96), 0.45, np.float32)
    depth[:24, :] = 0.92  # horizon benzeri uzak ust bant
    reconstructed = rgb.copy()

    # Ust bantta zayif yalanci sinyal
    rgb[8:18, 20:44] = 0.52
    reconstructed[8:18, 20:44] = 0.56

    # Uzak ama sahne ortasinda gercek hedef benzeri bolge
    rgb[48:60, 46:58] = 0.80
    reconstructed[48:60, 46:58] = 0.50
    depth[48:60, 46:58] = 0.82
    depth[46:62, 44:60] = 0.78

    combined, _ = compute_combined_anomaly_map(
        rgb,
        reconstructed,
        depth,
        hyst_high_pct=89,
        hyst_low_pct=78,
        top_k=20,
        edge_reinforce=0.65,
    )

    center_mean = float(combined[48:60, 46:58].mean())
    horizon_mean = float(combined[:24, :].mean())
    assert center_mean > 1.35 * horizon_mean
