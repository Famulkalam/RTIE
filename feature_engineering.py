"""
RTIE — Physics-Based Thermal Feature Engineering

Extracts 10 thermodynamic features from a radiator temperature matrix.
These features are fused with EfficientNet-B0 embeddings in the model.

Features:
    1. mean_temp          — Average temperature
    2. std_temp           — Temperature standard deviation
    3. max_temp           — Maximum temperature
    4. min_temp           — Minimum temperature
    5. left_right_diff    — Mean(left half) - Mean(right half)
    6. top_bottom_gradient — Mean(top 25%) - Mean(bottom 25%)
    7. cold_spot_area_ratio — Fraction of pixels < (mean - 1.5σ)
    8. entropy            — Shannon entropy of temperature distribution
    9. uniformity_index   — 1 - (std/mean), higher = more uniform
   10. hot_cold_ratio     — Fraction above 65°C / Fraction below 50°C
"""

import numpy as np
from scipy.stats import entropy as shannon_entropy
import config


def extract_features(matrix: np.ndarray) -> np.ndarray:
    """
    Extract 10 physics-based features from a temperature matrix.

    Args:
        matrix: 2D numpy array (rows × cols) of temperature values in °C.

    Returns:
        1D numpy array of 10 features.
    """
    rows, cols = matrix.shape
    flat = matrix.flatten()

    # 1-4. Basic statistics
    mean_temp = float(np.mean(flat))
    std_temp = float(np.std(flat))
    max_temp = float(np.max(flat))
    min_temp = float(np.min(flat))

    # 5. Left-right difference (imbalance indicator)
    mid_col = cols // 2
    left_mean = float(np.mean(matrix[:, :mid_col]))
    right_mean = float(np.mean(matrix[:, mid_col:]))
    left_right_diff = left_mean - right_mean

    # 6. Top-bottom gradient (blockage / air trapped indicator)
    top_quarter = int(rows * 0.25)
    bottom_quarter = int(rows * 0.75)
    top_mean = float(np.mean(matrix[:top_quarter, :]))
    bottom_mean = float(np.mean(matrix[bottom_quarter:, :]))
    top_bottom_gradient = top_mean - bottom_mean

    # 7. Cold spot area ratio
    cold_threshold = mean_temp - config.COLD_SPOT_THRESHOLD_SIGMA * std_temp
    cold_pixels = np.sum(flat < cold_threshold)
    cold_spot_area_ratio = float(cold_pixels / len(flat))

    # 8. Shannon entropy of temperature distribution
    # Bin temperatures into 50 bins
    hist, _ = np.histogram(flat, bins=50, density=True)
    hist = hist[hist > 0]  # remove zero bins
    temp_entropy = float(shannon_entropy(hist, base=2))

    # 9. Uniformity index (higher = more uniform heating)
    if mean_temp > 0:
        uniformity_index = float(1.0 - (std_temp / mean_temp))
    else:
        uniformity_index = 0.0

    # 10. Hot-cold ratio
    hot_fraction = float(np.sum(flat > config.HOT_THRESHOLD) / len(flat))
    cold_fraction = float(np.sum(flat < config.COLD_THRESHOLD) / len(flat))
    if cold_fraction > 0:
        hot_cold_ratio = hot_fraction / cold_fraction
    else:
        hot_cold_ratio = hot_fraction * 100 if hot_fraction > 0 else 1.0

    features = np.array([
        mean_temp,
        std_temp,
        max_temp,
        min_temp,
        left_right_diff,
        top_bottom_gradient,
        cold_spot_area_ratio,
        temp_entropy,
        uniformity_index,
        hot_cold_ratio,
    ], dtype=np.float32)

    return features


def feature_names() -> list:
    """Return ordered list of feature names."""
    return [
        "mean_temp",
        "std_temp",
        "max_temp",
        "min_temp",
        "left_right_diff",
        "top_bottom_gradient",
        "cold_spot_area_ratio",
        "entropy",
        "uniformity_index",
        "hot_cold_ratio",
    ]


if __name__ == "__main__":
    # Quick sanity check
    rng = np.random.default_rng(42)
    matrix = rng.uniform(50, 70, (config.GRID_ROWS, config.GRID_COLS))
    feats = extract_features(matrix)
    names = feature_names()
    print("Feature extraction test:")
    for name, val in zip(names, feats):
        print(f"  {name:25s} = {val:.4f}")
