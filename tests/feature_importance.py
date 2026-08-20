import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

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

BATCH_SIZE = 256

# Number of test sequences used for permutation analysis.
# Increase to 10000+ later for the final research experiment.
MAX_SAMPLES = 5000

OUTPUT_DIR = "results/feature_importance"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# Reproducibility
# ==========================================================

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)


# ==========================================================
# Device
# ==========================================================

print("=" * 70)
print("QUANTUM TRANSFORMER FEATURE IMPORTANCE")
print("=" * 70)

print(
    f"\nDevice : {DEVICE}"
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

print(
    f"Total test sequences : {len(dataset)}"
)


# ==========================================================
# Select test subset
# ==========================================================

num_samples = min(
    MAX_SAMPLES,
    len(dataset)
)

indices = np.arange(
    num_samples
)

test_subset = Subset(
    dataset,
    indices
)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

print(
    f"Samples used : {num_samples}"
)


# ==========================================================
# Feature names
# ==========================================================

feature_names = list(
    dataset.df.drop(
        columns=[
            "true_detuning",
            "time"
        ]
    ).columns
)

num_features = len(
    feature_names
)

print(
    f"Number of features : {num_features}"
)

print("\nFeatures:")
for i, name in enumerate(
    feature_names
):
    print(
        f"{i:02d} | {name}"
    )


# ==========================================================
# Load trained Transformer
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
    "\n✓ Trained Transformer loaded."
)


# ==========================================================
# Collect test data
# ==========================================================

all_x = []
all_y = []

print(
    "\nLoading test sequences..."
)

with torch.no_grad():

    for x, y in test_loader:

        all_x.append(
            x.cpu()
        )

        all_y.append(
            y.cpu()
        )


X = torch.cat(
    all_x,
    dim=0
)

Y = torch.cat(
    all_y,
    dim=0
)

print(
    f"Input tensor : {X.shape}"
)

print(
    f"Target tensor: {Y.shape}"
)


# ==========================================================
# Prediction helper
# ==========================================================

@torch.no_grad()
def predict(model, X):

    predictions = []

    loader = DataLoader(
        X,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    for batch in loader:

        batch = batch.to(
            DEVICE,
            non_blocking=True
        )

        prediction = model(
            batch
        )

        predictions.append(
            prediction.cpu()
        )

    return torch.cat(
        predictions
    )


# ==========================================================
# RMSE
# ==========================================================

def calculate_rmse(
    predictions,
    targets
):

    error = (
        predictions -
        targets
    )

    return torch.sqrt(
        torch.mean(
            error ** 2
        )
    ).item()


# ==========================================================
# Baseline prediction
# ==========================================================

print(
    "\nCalculating baseline performance..."
)

baseline_prediction = predict(
    model,
    X
)

baseline_rmse = calculate_rmse(
    baseline_prediction,
    Y
)

print(
    f"Baseline RMSE : "
    f"{baseline_rmse:.12e}"
)


# ==========================================================
# Permutation Feature Importance
# ==========================================================

importance_results = []

print(
    "\n" + "=" * 70
)

print(
    "PERMUTATION FEATURE IMPORTANCE"
)

print(
    "=" * 70
)


for feature_index, feature_name in enumerate(
    feature_names
):

    print(
        f"\n[{feature_index + 1:02d}/{num_features:02d}] "
        f"{feature_name}"
    )

    # ------------------------------------------------------
    # Clone original test tensor
    # ------------------------------------------------------

    X_permuted = X.clone()

    # ------------------------------------------------------
    # Shuffle complete temporal trajectories
    #
    # Important:
    #
    # We shuffle the feature between sequences rather than
    # independently shuffling individual timesteps.
    #
    # Therefore the temporal structure of the feature remains
    # intact.
    # ------------------------------------------------------

    permutation = rng.permutation(
        num_samples
    )

    permutation = torch.tensor(
        permutation,
        dtype=torch.long
    )

    X_permuted[:, :, feature_index] = (
        X[
            permutation,
            :,
            feature_index
        ]
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    permuted_prediction = predict(
        model,
        X_permuted
    )

    # ------------------------------------------------------
    # RMSE
    # ------------------------------------------------------

    permuted_rmse = calculate_rmse(
        permuted_prediction,
        Y
    )

    # ------------------------------------------------------
    # Importance
    # ------------------------------------------------------

    importance = (
        permuted_rmse -
        baseline_rmse
    )

    relative_importance = (
        importance /
        baseline_rmse
    ) * 100.0

    importance_results.append({

        "feature":
            feature_name,

        "baseline_rmse":
            baseline_rmse,

        "permuted_rmse":
            permuted_rmse,

        "importance":
            importance,

        "relative_importance_percent":
            relative_importance,

    })

    print(
        f"    Baseline RMSE : "
        f"{baseline_rmse:.8e}"
    )

    print(
        f"    Permuted RMSE : "
        f"{permuted_rmse:.8e}"
    )

    print(
        f"    Importance    : "
        f"{importance:+.8e}"
    )

    print(
        f"    Relative      : "
        f"{relative_importance:+.2f}%"
    )


# ==========================================================
# DataFrame
# ==========================================================

importance_df = pd.DataFrame(
    importance_results
)


# Sort by importance

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

# Rank

importance_df.insert(
    0,
    "rank",
    np.arange(
        1,
        len(importance_df) + 1
    )
)


# ==========================================================
# Normalized importance
# ==========================================================

positive_importance = (
    importance_df["importance"]
    .clip(lower=0)
)

total_importance = (
    positive_importance.sum()
)

if total_importance > 0:

    importance_df[
        "normalized_importance_percent"
    ] = (
        positive_importance /
        total_importance
    ) * 100.0

else:

    importance_df[
        "normalized_importance_percent"
    ] = 0.0


# ==========================================================
# Save CSV
# ==========================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

importance_df.to_csv(
    csv_path,
    index=False
)


# ==========================================================
# Print ranking
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "FEATURE IMPORTANCE RANKING"
)

print(
    "=" * 70
)

for _, row in importance_df.iterrows():

    print(
        f"{int(row['rank']):02d}. "
        f"{row['feature']:<30} "
        f"ΔRMSE = "
        f"{row['importance']:+.8e} "
        f"({row['relative_importance_percent']:+.2f}%)"
    )


# ==========================================================
# Plot 1 — Raw permutation importance
# ==========================================================

plot_df = importance_df.sort_values(
    "importance",
    ascending=True
)

plt.figure(
    figsize=(11, 8)
)

plt.barh(
    plot_df["feature"],
    plot_df["importance"]
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel(
    "Increase in RMSE after permutation"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Quantum Transformer Feature Importance"
)

plt.tight_layout()

raw_plot = os.path.join(
    OUTPUT_DIR,
    "feature_importance.png"
)

plt.savefig(
    raw_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# Plot 2 — Normalized importance
# ==========================================================

normalized_df = importance_df.sort_values(
    "normalized_importance_percent",
    ascending=True
)

plt.figure(
    figsize=(11, 8)
)

plt.barh(
    normalized_df["feature"],
    normalized_df[
        "normalized_importance_percent"
    ]
)

plt.xlabel(
    "Normalized Importance (%)"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Quantum Transformer — Normalized Feature Importance"
)

plt.tight_layout()

normalized_plot = os.path.join(
    OUTPUT_DIR,
    "feature_importance_normalized.png"
)

plt.savefig(
    normalized_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# Top features
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 10 FEATURES"
)

print(
    "=" * 70
)

for _, row in importance_df.head(
    10
).iterrows():

    print(
        f"{int(row['rank']):02d}. "
        f"{row['feature']:<30} "
        f"{row['normalized_importance_percent']:.2f}%"
    )


# ==========================================================
# Final
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "FEATURE IMPORTANCE ANALYSIS COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nBaseline RMSE : "
    f"{baseline_rmse:.12e}"
)

print(
    f"\nSaved:"
)

print(
    f"{csv_path}"
)

print(
    f"{raw_plot}"
)

print(
    f"{normalized_plot}"
)

print(
    "\n[PASS] Feature importance analysis completed."
)