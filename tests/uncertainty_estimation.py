import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

TEST_FILE = (
    "data/processed/clock_test_v3_scaled.csv"
)

SEQUENCE_LENGTH = 128

BATCH_SIZE = 64

MC_SAMPLES = 50

MAX_SAMPLES = 2000

OUTPUT_DIR = (
    "results/uncertainty"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==========================================================
# Header
# ==========================================================

print("=" * 70)
print("QUANTUM TRANSFORMER UNCERTAINTY ESTIMATION")
print("=" * 70)

print(
    f"\nDevice       : {DEVICE}"
)

print(
    f"MC samples   : {MC_SAMPLES}"
)


# ==========================================================
# Dataset
# ==========================================================

dataset = QuantumClockDataset(
    TEST_FILE,
    sequence_length=SEQUENCE_LENGTH,
    target_column="true_detuning",
    time_column="time",
)

num_samples = min(
    MAX_SAMPLES,
    len(dataset)
)

# ----------------------------------------------------------
# IMPORTANT:
# QuantumClockDataset creates the 128-step sequence inside
# __getitem__(). Therefore we must access the Dataset itself
# rather than dataset.features directly.
# ----------------------------------------------------------

X_list = []
Y_list = []

for i in range(num_samples):

    x, y = dataset[i]

    X_list.append(x)
    Y_list.append(y)

X = torch.stack(
    X_list
)

Y = torch.stack(
    Y_list
)

print(
    f"Test samples : {num_samples}"
)

print(
    f"Input shape  : {X.shape}"
)

print(
    f"Target shape : {Y.shape}"
)


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
# Enable dropout while keeping LayerNorm etc. in eval mode
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
    "✓ Monte Carlo dropout enabled."
)


# ==========================================================
# MC prediction
# ==========================================================

def mc_predict(
    model,
    X,
    mc_samples,
):

    predictions = []

    loader = DataLoader(
        X,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    for _ in range(mc_samples):

        batch_predictions = []

        for batch in loader:

            batch = batch.to(
                DEVICE,
                non_blocking=True
            )

            with torch.no_grad():

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

    return np.stack(
        predictions,
        axis=0
    )


# ==========================================================
# Run MC inference
# ==========================================================

print(
    "\nRunning Monte Carlo inference..."
)

mc_predictions = mc_predict(
    model,
    X,
    MC_SAMPLES,
)

print(
    f"MC prediction shape : "
    f"{mc_predictions.shape}"
)


# ==========================================================
# Statistics
# ==========================================================

mean_prediction = (
    mc_predictions.mean(
        axis=0
    )
)

std_prediction = (
    mc_predictions.std(
        axis=0
    )
)

lower_95 = (
    mean_prediction -
    1.96 * std_prediction
)

upper_95 = (
    mean_prediction +
    1.96 * std_prediction
)


targets = Y.numpy()

errors = (
    mean_prediction -
    targets
)

absolute_errors = (
    np.abs(errors)
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
# Coverage
# ==========================================================

inside_interval = (
    (targets >= lower_95)
    &
    (targets <= upper_95)
)

coverage = (
    np.mean(
        inside_interval
    )
    * 100.0
)

average_uncertainty = (
    np.mean(
        std_prediction
    )
)

median_uncertainty = (
    np.median(
        std_prediction
    )
)

max_uncertainty = (
    np.max(
        std_prediction
    )
)


# ==========================================================
# Results
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "UNCERTAINTY RESULTS"
)

print(
    "=" * 70
)

print(
    f"RMSE                    : "
    f"{rmse:.12e} Hz"
)

print(
    f"MAE                     : "
    f"{mae:.12e} Hz"
)

print(
    f"Mean uncertainty        : "
    f"{average_uncertainty:.12e} Hz"
)

print(
    f"Median uncertainty      : "
    f"{median_uncertainty:.12e} Hz"
)

print(
    f"Maximum uncertainty     : "
    f"{max_uncertainty:.12e} Hz"
)

print(
    f"95% interval coverage   : "
    f"{coverage:.2f}%"
)


# ==========================================================
# Save per-sample results
# ==========================================================

results_df = pd.DataFrame({

    "target":
        targets,

    "prediction":
        mean_prediction,

    "uncertainty_std":
        std_prediction,

    "lower_95":
        lower_95,

    "upper_95":
        upper_95,

    "absolute_error":
        absolute_errors,

    "inside_95_interval":
        inside_interval,

})


csv_path = os.path.join(
    OUTPUT_DIR,
    "uncertainty_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ==========================================================
# Summary
# ==========================================================

summary_df = pd.DataFrame({

    "metric": [
        "RMSE_Hz",
        "MAE_Hz",
        "Mean_uncertainty_Hz",
        "Median_uncertainty_Hz",
        "Maximum_uncertainty_Hz",
        "Coverage_percent",
        "MC_samples",
        "Test_samples",
    ],

    "value": [
        rmse,
        mae,
        average_uncertainty,
        median_uncertainty,
        max_uncertainty,
        coverage,
        MC_SAMPLES,
        num_samples,
    ],

})


summary_path = os.path.join(
    OUTPUT_DIR,
    "uncertainty_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ==========================================================
# Plot 1 — Prediction intervals
# ==========================================================

N_PLOT = min(
    500,
    num_samples
)

x_axis = np.arange(
    N_PLOT
)

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    x_axis,
    targets[:N_PLOT],
    label="True Detuning"
)

plt.plot(
    x_axis,
    mean_prediction[:N_PLOT],
    label="Prediction"
)

plt.fill_between(
    x_axis,
    lower_95[:N_PLOT],
    upper_95[:N_PLOT],
    alpha=0.25,
    label="95% Prediction Interval"
)

plt.xlabel(
    "Test Sample"
)

plt.ylabel(
    "Detuning (Hz)"
)

plt.title(
    "Quantum Transformer Prediction "
    "with Uncertainty"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "prediction_uncertainty.png"
    ),
    dpi=300
)

plt.close()


# ==========================================================
# Plot 2 — Uncertainty distribution
# ==========================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    std_prediction,
    bins=50
)

plt.xlabel(
    "Predictive Standard Deviation (Hz)"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Transformer Predictive Uncertainty Distribution"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "uncertainty_distribution.png"
    ),
    dpi=300
)

plt.close()


# ==========================================================
# Plot 3 — Error vs uncertainty
# ==========================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    std_prediction,
    absolute_errors,
    s=8,
    alpha=0.5
)

plt.xlabel(
    "Predictive Uncertainty (Hz)"
)

plt.ylabel(
    "Absolute Prediction Error (Hz)"
)

plt.title(
    "Prediction Error vs Model Uncertainty"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "error_vs_uncertainty.png"
    ),
    dpi=300
)

plt.close()


# ==========================================================
# Final
# ==========================================================

print(
    "\nSaved:"
)

print(
    f"{csv_path}"
)

print(
    f"{summary_path}"
)

print(
    f"{OUTPUT_DIR}/prediction_uncertainty.png"
)

print(
    f"{OUTPUT_DIR}/uncertainty_distribution.png"
)

print(
    f"{OUTPUT_DIR}/error_vs_uncertainty.png"
)

print(
    "\n[PASS] Uncertainty estimation completed."
)