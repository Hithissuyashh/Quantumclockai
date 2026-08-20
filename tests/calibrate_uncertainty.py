import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.config import CHECKPOINT_DIR
from ml.datasets.sequence_dataset import QuantumClockDataset
from ml.models.transformer_model import QuantumTransformer


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_FILE = "data/processed/clock_test_v3_scaled.csv"

SEQUENCE_LENGTH = 128
BATCH_SIZE = 64
MC_SAMPLES = 50

TOTAL_SAMPLES = 2000

# 50% calibration / 50% independent evaluation
CALIBRATION_SAMPLES = TOTAL_SAMPLES // 2
EVALUATION_SAMPLES = TOTAL_SAMPLES - CALIBRATION_SAMPLES

OUTPUT_DIR = "results/uncertainty"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================================
# Header
# ==========================================================

print("=" * 70)
print("MC DROPOUT UNCERTAINTY CALIBRATION")
print("=" * 70)

print(f"\nDevice       : {DEVICE}")
print(f"MC samples   : {MC_SAMPLES}")
print(f"Total samples: {TOTAL_SAMPLES}")
print(f"Calibration  : {CALIBRATION_SAMPLES}")
print(f"Evaluation   : {EVALUATION_SAMPLES}")


# ==========================================================
# Dataset
# ==========================================================

dataset = QuantumClockDataset(
    TEST_FILE,
    sequence_length=SEQUENCE_LENGTH,
    target_column="true_detuning",
    time_column="time",
)

total_available = len(dataset)

num_samples = min(
    TOTAL_SAMPLES,
    total_available
)

print(
    f"\nAvailable sequences : {total_available}"
)
print(
    f"Using sequences     : {num_samples}"
)


# ==========================================================
# Build sequences correctly through Dataset.__getitem__
# ==========================================================

X_list = []
Y_list = []

for i in range(num_samples):

    x, y = dataset[i]

    X_list.append(x)
    Y_list.append(y)


X = torch.stack(X_list)
Y = torch.stack(Y_list)

print(
    f"Input shape  : {X.shape}"
)

print(
    f"Target shape : {Y.shape}"
)


# ==========================================================
# Calibration / evaluation split
# ==========================================================

calibration_X = X[
    :CALIBRATION_SAMPLES
]

calibration_Y = Y[
    :CALIBRATION_SAMPLES
]

evaluation_X = X[
    CALIBRATION_SAMPLES:
]

evaluation_Y = Y[
    CALIBRATION_SAMPLES:
]


# ==========================================================
# Load model
# ==========================================================

model = QuantumTransformer().to(
    DEVICE
)

checkpoint = torch.load(
    CHECKPOINT_DIR / "best_model.pth",
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    "✓ Trained Transformer loaded."
)


# ==========================================================
# Enable MC Dropout
# ==========================================================

def enable_dropout(module):

    if isinstance(
        module,
        nn.Dropout
    ):
        module.train()


model.eval()

model.apply(
    enable_dropout
)

print(
    "✓ MC Dropout enabled."
)


# ==========================================================
# MC prediction
# ==========================================================

@torch.no_grad()
def mc_predict(
    model,
    X,
    mc_samples
):

    predictions = []

    loader = DataLoader(
        X,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    for iteration in range(
        mc_samples
    ):

        batch_predictions = []

        for batch in loader:

            batch = batch.to(
                DEVICE,
                non_blocking=True
            )

            prediction = model(
                batch
            )

            batch_predictions.append(
                prediction.cpu()
            )

        predictions.append(
            torch.cat(
                batch_predictions
            ).numpy()
        )

        if (
            iteration + 1
        ) % 10 == 0:

            print(
                f"MC pass "
                f"{iteration + 1:02d}/"
                f"{mc_samples}"
            )

    return np.stack(
        predictions,
        axis=0
    )


# ==========================================================
# Calibration MC inference
# ==========================================================

print(
    "\nRunning calibration inference..."
)

calibration_mc = mc_predict(
    model,
    calibration_X,
    MC_SAMPLES
)

calibration_mean = (
    calibration_mc.mean(
        axis=0
    )
)

calibration_std = (
    calibration_mc.std(
        axis=0
    )
)

calibration_targets = (
    calibration_Y.numpy()
)


# ==========================================================
# Calculate nonconformity scores
#
# We need:
#
# |error| / predicted_std
#
# The 95th percentile becomes the empirical
# calibration multiplier.
# ==========================================================

calibration_error = np.abs(
    calibration_mean -
    calibration_targets
)

# Avoid division by zero
safe_std = np.maximum(
    calibration_std,
    1e-12
)

nonconformity = (
    calibration_error /
    safe_std
)

calibration_factor = np.quantile(
    nonconformity,
    0.95
)


# ==========================================================
# Raw calibration coverage
# ==========================================================

raw_lower = (
    calibration_mean -
    1.96 * calibration_std
)

raw_upper = (
    calibration_mean +
    1.96 * calibration_std
)

raw_inside = (
    (calibration_targets >= raw_lower)
    &
    (calibration_targets <= raw_upper)
)

raw_coverage = (
    np.mean(raw_inside)
    * 100.0
)


# ==========================================================
# Calibrated calibration coverage
# ==========================================================

cal_lower = (
    calibration_mean -
    calibration_factor *
    calibration_std
)

cal_upper = (
    calibration_mean +
    calibration_factor *
    calibration_std
)

cal_inside = (
    (calibration_targets >= cal_lower)
    &
    (calibration_targets <= cal_upper)
)

calibrated_coverage = (
    np.mean(cal_inside)
    * 100.0
)


# ==========================================================
# Independent evaluation MC inference
# ==========================================================

print(
    "\nRunning independent evaluation inference..."
)

evaluation_mc = mc_predict(
    model,
    evaluation_X,
    MC_SAMPLES
)

evaluation_mean = (
    evaluation_mc.mean(
        axis=0
    )
)

evaluation_std = (
    evaluation_mc.std(
        axis=0
    )
)

evaluation_targets = (
    evaluation_Y.numpy()
)


# ==========================================================
# Evaluation metrics
# ==========================================================

errors = (
    evaluation_mean -
    evaluation_targets
)

absolute_errors = np.abs(
    errors
)

rmse = np.sqrt(
    np.mean(
        errors ** 2
    )
)

mae = np.mean(
    absolute_errors
)


# ==========================================================
# Raw evaluation interval
# ==========================================================

raw_eval_lower = (
    evaluation_mean -
    1.96 * evaluation_std
)

raw_eval_upper = (
    evaluation_mean +
    1.96 * evaluation_std
)

raw_eval_inside = (
    (evaluation_targets >= raw_eval_lower)
    &
    (evaluation_targets <= raw_eval_upper)
)

raw_eval_coverage = (
    np.mean(raw_eval_inside)
    * 100.0
)


# ==========================================================
# Calibrated evaluation interval
# ==========================================================

cal_eval_lower = (
    evaluation_mean -
    calibration_factor *
    evaluation_std
)

cal_eval_upper = (
    evaluation_mean +
    calibration_factor *
    evaluation_std
)

cal_eval_inside = (
    (evaluation_targets >= cal_eval_lower)
    &
    (evaluation_targets <= cal_eval_upper)
)

cal_eval_coverage = (
    np.mean(cal_eval_inside)
    * 100.0
)


# ==========================================================
# Uncertainty statistics
# ==========================================================

mean_raw_uncertainty = (
    np.mean(
        evaluation_std
    )
)

mean_calibrated_uncertainty = (
    calibration_factor *
    mean_raw_uncertainty
)


# ==========================================================
# Results
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "UNCERTAINTY CALIBRATION RESULTS"
)

print(
    "=" * 70
)

print(
    f"Raw calibration coverage       : "
    f"{raw_coverage:.2f}%"
)

print(
    f"Calibrated calibration coverage: "
    f"{calibrated_coverage:.2f}%"
)

print(
    f"\nCalibration factor             : "
    f"{calibration_factor:.6f}"
)

print(
    "\nIndependent evaluation"
)

print(
    f"RMSE                           : "
    f"{rmse:.12e} Hz"
)

print(
    f"MAE                            : "
    f"{mae:.12e} Hz"
)

print(
    f"Mean raw uncertainty           : "
    f"{mean_raw_uncertainty:.12e} Hz"
)

print(
    f"Mean calibrated uncertainty    : "
    f"{mean_calibrated_uncertainty:.12e} Hz"
)

print(
    f"\nRaw 95% interval coverage      : "
    f"{raw_eval_coverage:.2f}%"
)

print(
    f"Calibrated interval coverage   : "
    f"{cal_eval_coverage:.2f}%"
)


# ==========================================================
# Save evaluation results
# ==========================================================

results_df = pd.DataFrame({

    "target":
        evaluation_targets,

    "prediction":
        evaluation_mean,

    "raw_uncertainty":
        evaluation_std,

    "calibrated_uncertainty":
        calibration_factor *
        evaluation_std,

    "raw_lower_95":
        raw_eval_lower,

    "raw_upper_95":
        raw_eval_upper,

    "calibrated_lower":
        cal_eval_lower,

    "calibrated_upper":
        cal_eval_upper,

    "absolute_error":
        absolute_errors,

    "raw_inside_interval":
        raw_eval_inside,

    "calibrated_inside_interval":
        cal_eval_inside,

})

results_path = os.path.join(
    OUTPUT_DIR,
    "calibrated_uncertainty_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ==========================================================
# Save summary
# ==========================================================

summary_df = pd.DataFrame({

    "metric": [

        "RMSE_Hz",
        "MAE_Hz",

        "Raw_calibration_coverage_percent",
        "Calibrated_calibration_coverage_percent",

        "Calibration_factor",

        "Raw_evaluation_coverage_percent",
        "Calibrated_evaluation_coverage_percent",

        "Mean_raw_uncertainty_Hz",
        "Mean_calibrated_uncertainty_Hz",

        "MC_samples",
        "Calibration_samples",
        "Evaluation_samples",

    ],

    "value": [

        rmse,
        mae,

        raw_coverage,
        calibrated_coverage,

        calibration_factor,

        raw_eval_coverage,
        cal_eval_coverage,

        mean_raw_uncertainty,
        mean_calibrated_uncertainty,

        MC_SAMPLES,
        CALIBRATION_SAMPLES,
        EVALUATION_SAMPLES,

    ]

})


summary_path = os.path.join(
    OUTPUT_DIR,
    "calibrated_uncertainty_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ==========================================================
# Final
# ==========================================================

print(
    "\nSaved:"
)

print(
    results_path
)

print(
    summary_path
)

print(
    "\n[PASS] Uncertainty calibration completed."
)