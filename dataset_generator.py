"""
RTIE — Synthetic Radiator Thermal Dataset Generator

Generates 5,000 thermal heatmap images (1,000 per class) with physics-based
temperature patterns and noise injection for realism.

Classes:
    - efficient:    Smooth vertical gradient (70°C top → 60°C bottom)
    - imbalance:    Left-right asymmetry (70°C left, 50°C right)
    - blockage:     Cold bottom zone (bottom 25% ~40°C)
    - scaling:      Random clustered cold patches
    - air_trapped:  Cold top-right corner region
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import config
from feature_engineering import extract_features


# ─────────────────── Temperature Matrix Generators ───────────────────

def _generate_efficient(rows, cols, rng):
    """Smooth vertical gradient: top 70°C → bottom 60°C, uniform lateral."""
    top_temp = rng.uniform(68, 72)
    bottom_temp = rng.uniform(58, 62)
    gradient = np.linspace(top_temp, bottom_temp, rows).reshape(-1, 1)
    matrix = np.tile(gradient, (1, cols))
    # Add slight natural variation
    matrix += rng.normal(0, 0.3, (rows, cols))
    return matrix


def _generate_imbalance(rows, cols, rng):
    """Left side hot (~70°C), right side cold (~50°C)."""
    left_temp = rng.uniform(67, 73)
    right_temp = rng.uniform(47, 53)
    lateral = np.linspace(left_temp, right_temp, cols).reshape(1, -1)
    # Add vertical gradient too (top slightly hotter)
    vertical = np.linspace(1.0, 0.95, rows).reshape(-1, 1)
    matrix = np.tile(lateral, (rows, 1)) * vertical
    matrix += rng.normal(0, 0.5, (rows, cols))
    return matrix


def _generate_blockage(rows, cols, rng):
    """Normal top, cold bottom 25% (~40°C) due to sludge blockage."""
    top_temp = rng.uniform(68, 72)
    bottom_temp = rng.uniform(58, 62)
    gradient = np.linspace(top_temp, bottom_temp, rows).reshape(-1, 1)
    matrix = np.tile(gradient, (1, cols))

    # Cold bottom zone
    blockage_start = int(rows * rng.uniform(0.70, 0.80))
    cold_temp = rng.uniform(38, 44)
    transition = np.linspace(matrix[blockage_start, 0], cold_temp, rows - blockage_start)
    matrix[blockage_start:, :] = transition.reshape(-1, 1)
    matrix += rng.normal(0, 0.4, (rows, cols))
    return matrix


def _generate_scaling(rows, cols, rng):
    """Base gradient with clustered cold patches (limescale deposits)."""
    top_temp = rng.uniform(68, 72)
    bottom_temp = rng.uniform(58, 62)
    gradient = np.linspace(top_temp, bottom_temp, rows).reshape(-1, 1)
    matrix = np.tile(gradient, (1, cols))

    # Generate Perlin-like noise for cold patches
    num_patches = rng.integers(3, 8)
    for _ in range(num_patches):
        cy, cx = rng.integers(5, rows - 5), rng.integers(3, cols - 3)
        ry, rx = rng.integers(3, 8), rng.integers(2, 6)
        y_range = slice(max(0, cy - ry), min(rows, cy + ry))
        x_range = slice(max(0, cx - rx), min(cols, cx + rx))
        patch = matrix[y_range, x_range]
        cold_drop = rng.uniform(8, 18)
        # Smooth cold patch
        cold_mask = np.ones_like(patch) * cold_drop
        cold_mask = gaussian_filter(cold_mask, sigma=2.0)
        matrix[y_range, x_range] -= cold_mask

    matrix += rng.normal(0, 0.5, (rows, cols))
    return matrix


def _generate_air_trapped(rows, cols, rng):
    """Cold top-right corner where air pocket displaces water."""
    top_temp = rng.uniform(68, 72)
    bottom_temp = rng.uniform(58, 62)
    gradient = np.linspace(top_temp, bottom_temp, rows).reshape(-1, 1)
    matrix = np.tile(gradient, (1, cols))

    # Cold top-right region
    cold_temp = rng.uniform(40, 45)
    corner_rows = int(rows * rng.uniform(0.20, 0.35))
    corner_cols = int(cols * rng.uniform(0.30, 0.50))

    for r in range(corner_rows):
        for c in range(cols - corner_cols, cols):
            dist = np.sqrt((r / corner_rows) ** 2 + ((c - (cols - corner_cols)) / corner_cols) ** 2)
            blend = max(0, 1 - dist)
            matrix[r, c] = matrix[r, c] * (1 - blend) + cold_temp * blend

    matrix += rng.normal(0, 0.4, (rows, cols))
    return matrix


GENERATORS = {
    "efficient": _generate_efficient,
    "imbalance": _generate_imbalance,
    "blockage": _generate_blockage,
    "scaling": _generate_scaling,
    "air_trapped": _generate_air_trapped,
}


# ─────────────────── Noise Injection ───────────────────

def inject_noise(matrix, rng):
    """Apply realistic sensor noise to temperature matrix."""
    rows, cols = matrix.shape

    # 1. Gaussian sensor noise
    sigma = rng.uniform(*config.NOISE_SIGMA_RANGE)
    matrix = matrix + rng.normal(0, sigma, (rows, cols))

    # 2. Global ambient temperature offset
    offset = rng.uniform(*config.AMBIENT_OFFSET_RANGE)
    matrix = matrix + offset

    # 3. Spatial smoothing jitter
    blur_sigma = rng.uniform(*config.BLUR_SIGMA_RANGE)
    matrix = gaussian_filter(matrix, sigma=blur_sigma)

    return matrix


# ─────────────────── Heatmap Rendering ───────────────────

def matrix_to_image(matrix, size=config.IMAGE_SIZE):
    """Convert temperature matrix to grayscale heatmap image (224×224)."""
    # Normalize to 0-255 range
    min_val, max_val = matrix.min(), matrix.max()
    if max_val - min_val < 1e-6:
        normalized = np.zeros_like(matrix)
    else:
        normalized = (matrix - min_val) / (max_val - min_val)
    normalized = (normalized * 255).astype(np.uint8)

    img = Image.fromarray(normalized, mode="L")
    img = img.resize((size, size), Image.BICUBIC)
    return img


def compute_efficiency_score(matrix, class_name):
    """Compute physics-based efficiency score (0-100)."""
    mean_temp = matrix.mean()
    std_temp = matrix.std()

    # Ideal: high mean, low std (uniform heat distribution)
    # Baseline: efficient radiator ~ 65°C mean, ~3°C std
    mean_score = min(100, max(0, (mean_temp - 30) / 40 * 100))
    uniformity_score = min(100, max(0, 100 - std_temp * 5))

    # Weight: 60% mean temp, 40% uniformity
    score = 0.6 * mean_score + 0.4 * uniformity_score

    # Apply class-specific penalty
    penalty = config.ENERGY_LOSS_MAP.get(class_name, 0)
    score = score * (1 - penalty / 100)

    return round(np.clip(score, 0, 100), 1)


# ─────────────────── Main Generation Pipeline ───────────────────

def generate_dataset():
    """Generate full synthetic radiator thermal dataset."""
    os.makedirs(config.SYNTHETIC_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    metadata = []
    total = config.NUM_CLASSES * config.SAMPLES_PER_CLASS

    print(f"Generating {total} synthetic thermal images...")
    print(f"  Classes: {config.CLASS_NAMES}")
    print(f"  Grid: {config.GRID_ROWS}×{config.GRID_COLS}")
    print(f"  Image: {config.IMAGE_SIZE}×{config.IMAGE_SIZE} grayscale PNG")
    print(f"  Noise injection: enabled")
    print()

    with tqdm(total=total, desc="Generating") as pbar:
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(config.SYNTHETIC_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)

            generator = GENERATORS[class_name]

            for i in range(config.SAMPLES_PER_CLASS):
                # Per-image reproducible seed
                seed = hash((class_name, i)) % (2**32)
                rng = np.random.default_rng(seed)

                # Generate base temperature matrix
                matrix = generator(config.GRID_ROWS, config.GRID_COLS, rng)

                # Inject noise
                matrix = inject_noise(matrix, rng)

                # Compute features before image conversion
                features = extract_features(matrix)

                # Compute efficiency score
                efficiency = compute_efficiency_score(matrix, class_name)

                # Save features
                feat_path = os.path.join(class_dir, f"{i:04d}_features.npy")
                np.save(feat_path, features)

                # Save temperature matrix
                mat_path = os.path.join(class_dir, f"{i:04d}_matrix.npy")
                np.save(mat_path, matrix)

                # Render and save image
                img = matrix_to_image(matrix)
                img_path = os.path.join(class_dir, f"{i:04d}.png")
                img.save(img_path)

                metadata.append({
                    "filename": f"{class_name}/{i:04d}.png",
                    "class": class_name,
                    "class_idx": config.CLASS_NAMES.index(class_name),
                    "efficiency_score": efficiency,
                    "mean_temp": round(float(matrix.mean()), 2),
                    "std_temp": round(float(matrix.std()), 2),
                })

                pbar.update(1)

    # Save metadata CSV
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(config.SYNTHETIC_DIR, "metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDataset generated successfully!")
    print(f"  Images: {config.SYNTHETIC_DIR}")
    print(f"  Metadata: {csv_path}")
    print(f"  Samples per class:")
    for cls in config.CLASS_NAMES:
        count = len(df[df["class"] == cls])
        print(f"    {cls}: {count}")

    return df


if __name__ == "__main__":
    generate_dataset()
