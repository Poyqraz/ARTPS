from src.models.depth_estimation import DEPTH_FEATURE_KEYS, MiDaSDepthEstimator


def test_vectorize_depth_features_uses_fixed_training_order():
    features = {
        "depth_gradient_max": 14.0,
        "depth_mean": 1.0,
        "depth_std": 2.0,
        "depth_min": 3.0,
        "depth_max": 4.0,
        "depth_median": 5.0,
        "depth_percentile_25": 6.0,
        "depth_percentile_75": 7.0,
        "depth_variance": 8.0,
        "depth_skewness": 9.0,
        "depth_kurtosis": 10.0,
        "surface_complexity": 11.0,
        "depth_gradient_mean": 12.0,
        "depth_gradient_std": 13.0,
        "roughness": 99.0,
    }

    vector = MiDaSDepthEstimator.vectorize_depth_features(features)

    assert len(vector) == len(DEPTH_FEATURE_KEYS) == 14
    assert vector == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]


def test_vectorize_depth_features_backfills_missing_values():
    vector = MiDaSDepthEstimator.vectorize_depth_features({"depth_mean": 0.25})

    assert len(vector) == len(DEPTH_FEATURE_KEYS)
    assert vector[0] == 0.25
    assert all(value == 0.0 for value in vector[1:])
