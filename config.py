"""
RTIE Configuration — Central configuration for all modules.
"""
import os

# ──────────────────────── Paths ────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
GRADCAM_DIR = os.path.join(REPORT_DIR, "gradcam")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ──────────────────────── Classes ────────────────────────
CLASS_NAMES = ["efficient", "imbalance", "blockage", "scaling", "air_trapped"]
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────── Synthetic Data Generation ────────────────────────
GRID_ROWS = 64
GRID_COLS = 32
IMAGE_SIZE = 224
SAMPLES_PER_CLASS = 1000

# Noise injection parameters
NOISE_SIGMA_RANGE = (0.5, 2.0)        # °C Gaussian sensor noise
AMBIENT_OFFSET_RANGE = (-1.5, 1.5)    # °C global temperature offset
BLUR_SIGMA_RANGE = (0.3, 0.8)         # spatial smoothing jitter

# ──────────────────────── Feature Engineering ────────────────────────
NUM_PHYSICS_FEATURES = 10
COLD_SPOT_THRESHOLD_SIGMA = 1.5       # pixels < (mean - 1.5σ) are cold spots
HOT_THRESHOLD = 65.0                  # °C
COLD_THRESHOLD = 50.0                 # °C

# ──────────────────────── Model ────────────────────────
BACKBONE = "efficientnet_b0"
BACKBONE_DIM = 1280
FUSION_DIM = BACKBONE_DIM + NUM_PHYSICS_FEATURES  # 1290
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
MC_DROPOUT_RATE = 0.3
MC_FORWARD_PASSES = 30

# ──────────────────────── Uncertainty Escalation ────────────────────────
UNCERTAINTY_STD_THRESHOLD = 0.15
CONFIDENCE_THRESHOLD = 0.70

# ──────────────────────── Training ────────────────────────
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 30
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5
LAMBDA_EFFICIENCY = 0.5
LAMBDA_CONFIDENCE = 0.3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────── Business Impact ────────────────────────
ENERGY_LOSS_MAP = {
    "efficient": 0.0,
    "imbalance": 12.0,
    "blockage": 18.0,
    "scaling": 9.0,
    "air_trapped": 15.0,
}

# Average UK radiator: ~1.5 kW, runs ~8h/day, ~180 days/year
RADIATOR_POWER_KW = 1.5
DAILY_HOURS = 8
HEATING_DAYS_PER_YEAR = 180
ENERGY_PRICE_GBP_PER_KWH = 0.28
CO2_KG_PER_KWH = 0.233  # UK grid average

# ──────────────────────── Robustness Testing ────────────────────────
ROBUSTNESS_NOISE_SIGMAS = [5, 10, 15]
ROBUSTNESS_TEMP_SHIFTS = [-5, -2, 2, 5]
ROBUSTNESS_BLUR_KERNELS = [3, 5, 7]
ROBUSTNESS_ROTATIONS = [-20, -10, 10, 20]

# ──────────────────────── Field Simulation ────────────────────────
FIELD_ASPECT_RATIOS = [(64, 48), (64, 24), (48, 48)]
FIELD_SCALE_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
FIELD_TEMP_SHIFTS = [-3, 0, 3]
FIELD_CONTRAST_SCALES = [0.9, 1.0, 1.1]
