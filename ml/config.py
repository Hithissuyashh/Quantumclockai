from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "clock_dataset_v2.csv"

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clock_dataset_scaled.csv"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ==========================================================
# Dataset
# ==========================================================

SEQUENCE_LENGTH = 128

NUM_FEATURES = 20

TARGET_COLUMN = "fractional_frequency"

# ==========================================================
# Training
# ==========================================================

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

TRAIN_SPLIT = 0.70

VAL_SPLIT = 0.15

TEST_SPLIT = 0.15

RANDOM_SEED = 42

# ==========================================================
# Transformer
# ==========================================================

D_MODEL = 64

NHEAD = 8

NUM_ENCODER_LAYERS = 4

DIM_FEEDFORWARD = 256

DROPOUT = 0.1

# ==========================================================
# Device
# ==========================================================

DEVICE = "cuda"