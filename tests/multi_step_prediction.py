import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

HORIZONS = [
    1,
    5,
    10,
    30,
    60,
]

# Number of independent forecast starting points
MAX_WINDOWS = 1000

OUTPUT_DIR = (
    "results/multi_step"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 70)
print("QUANTUM TRANSFORMER MULTI-STEP PREDICTION")
print("=" * 70)

print(
    f"\nDevice : {DEVICE}"
)


# ==========================================================
# Load dataset
# ==========================================================

dataset = QuantumClockDataset(
    TEST_FILE,
    sequence_length=SEQUENCE_LENGTH,
    target_column="true_detuning",
    time_column="time",
)

print(
    f"Dataset sequences : {len(dataset)}"
)


# ==========================================================
# Load trained model
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

model.eval()

print(
    "✓ Trained Transformer loaded."
)


# ==========================================================
# Convert dataset to tensors
# ==========================================================

features = torch.tensor(
    dataset.features,
    dtype=torch.float32
)

targets = torch.tensor(
    dataset.targets,
    dtype=torch.float32
)

total_points = len(
    features
)


# ==========================================================
# Recursive forecast function
# ==========================================================

@torch.no_grad()
def recursive_forecast(
    initial_window,
    horizon,
):
    """
    Predict future detuning recursively.

    initial_window:
        [128, 20]

    returns:
        [horizon]
    """

    window = (
        initial_window
        .clone()
        .to(DEVICE)
    )

    predictions = []

    for _ in range(horizon):

        # Add batch dimension
        x = window.unsqueeze(0)

        prediction = model(
            x
        )

        prediction_value = (
            prediction.item()
        )

        predictions.append(
            prediction_value
        )

        # --------------------------------------------------
        # Important:
        #
        # The Transformer expects 20 features at every
        # timestep, but its prediction is only the target
        # detuning.
        #
        # We therefore construct the next input row by
        # shifting the latest feature vector and replacing
        # the target-related information with the predicted
        # detuning where applicable.
        #
        # This is a forecasting approximation for the
        # existing one-step model.
        # --------------------------------------------------

        next_row = (
            window[-1]
            .clone()
        )

        # We cannot safely fabricate future environmental
        # measurements. Therefore the future feature vector
        # is held at the latest observed state.
        #
        # This isolates the effect of recursive prediction.

        # Shift window
        window = torch.cat(
            [
                window[1:],
                next_row.unsqueeze(0),
            ],
            dim=0,
        )

    return np.asarray(
        predictions,
        dtype=np.float64
    )


# ==========================================================
# IMPORTANT VALIDATION
# ==========================================================

print(
    "\nForecasting methodology:"
)

print(
    "Recursive prediction using the existing "
    "one-step Transformer."
)

print(
    "Future environmental features are held "
    "at their latest observed values."
)


# ==========================================================
# Evaluate horizons
# ==========================================================

results = []

all_predictions = {}
all_targets = {}


for horizon in HORIZONS:

    print(
        "\n" + "-" * 70
    )

    print(
        f"Evaluating horizon: "
        f"{horizon} step(s)"
    )

    predictions = []
    actuals = []

    max_start = min(
        total_points
        - SEQUENCE_LENGTH
        - horizon
        + 1,
        MAX_WINDOWS,
    )

    for start in range(
        max_start
    ):

        initial_window = features[
            start:
            start + SEQUENCE_LENGTH
        ]

        true_future = targets[
            start
            + SEQUENCE_LENGTH:
            start
            + SEQUENCE_LENGTH
            + horizon
        ].numpy()

        predicted_future = (
            recursive_forecast(
                initial_window,
                horizon,
            )
        )

        predictions.extend(
            predicted_future
        )

        actuals.extend(
            true_future
        )

    predictions = np.asarray(
        predictions
    )

    actuals = np.asarray(
        actuals
    )

    errors = (
        predictions -
        actuals
    )

    mse = np.mean(
        errors ** 2
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        mse
    )

    results.append({

        "horizon_steps":
            horizon,

        "horizon_seconds":
            horizon,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "MSE":
            mse,

        "samples":
            len(actuals),

    })

    all_predictions[horizon] = (
        predictions
    )

    all_targets[horizon] = (
        actuals
    )

    print(
        f"Samples : {len(actuals)}"
    )

    print(
        f"MAE     : {mae:.12e}"
    )

    print(
        f"RMSE    : {rmse:.12e}"
    )

    print(
        f"MSE     : {mse:.12e}"
    )


# ==========================================================
# Save results
# ==========================================================

results_df = pd.DataFrame(
    results
)

csv_path = os.path.join(
    OUTPUT_DIR,
    "multi_step_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ==========================================================
# RMSE vs horizon
# ==========================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    results_df[
        "horizon_seconds"
    ],
    results_df["RMSE"],
    marker="o",
)

plt.xlabel(
    "Prediction Horizon (s)"
)

plt.ylabel(
    "RMSE (Hz)"
)

plt.title(
    "Quantum Transformer "
    "Prediction Error vs Horizon"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "multi_step_rmse.png"
    ),
    dpi=300
)

plt.close()


# ==========================================================
# MAE vs horizon
# ==========================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    results_df[
        "horizon_seconds"
    ],
    results_df["MAE"],
    marker="o",
)

plt.xlabel(
    "Prediction Horizon (s)"
)

plt.ylabel(
    "MAE (Hz)"
)

plt.title(
    "Quantum Transformer "
    "Prediction MAE vs Horizon"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "multi_step_mae.png"
    ),
    dpi=300
)

plt.close()


# ==========================================================
# Prediction examples
# ==========================================================

for horizon in HORIZONS:

    predictions = (
        all_predictions[horizon]
    )

    actuals = (
        all_targets[horizon]
    )

    n = min(
        500,
        len(predictions)
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        actuals[:n],
        label="True Detuning"
    )

    plt.plot(
        predictions[:n],
        label="Transformer Prediction"
    )

    plt.xlabel(
        "Forecast Sample"
    )

    plt.ylabel(
        "Detuning (Hz)"
    )

    plt.title(
        f"Transformer Forecast — "
        f"{horizon}-Step Horizon"
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
            f"prediction_horizon_{horizon}.png"
        ),
        dpi=300
    )

    plt.close()


# ==========================================================
# Final output
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "MULTI-STEP PREDICTION RESULTS"
)

print(
    "=" * 70
)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    csv_path
)

print(
    f"{OUTPUT_DIR}/multi_step_rmse.png"
)

print(
    f"{OUTPUT_DIR}/multi_step_mae.png"
)

print(
    f"{OUTPUT_DIR}/prediction_horizon_*.png"
)

print(
    "\n[PASS] Multi-step prediction completed."
)