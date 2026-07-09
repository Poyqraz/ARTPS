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
