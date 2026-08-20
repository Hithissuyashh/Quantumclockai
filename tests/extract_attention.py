import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from ml.config import (
    CHECKPOINT_DIR,
    D_MODEL,
    NHEAD,
    NUM_ENCODER_LAYERS,
    DIM_FEEDFORWARD,
    DROPOUT,
)

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

OUTPUT_DIR = "results/attention"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# Attention-enabled Transformer Encoder Layer
# ==========================================================

class AttentionEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
    ):

        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(
            d_model,
            dim_feedforward,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.linear2 = nn.Linear(
            dim_feedforward,
            d_model,
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

        self.activation = nn.GELU()

        self.attention_weights = None

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=False,
    ):

        # --------------------------------------------------
        # Self attention
        # --------------------------------------------------

        attention_output, attention_weights = (
            self.self_attn(
                src,
                src,
                src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=is_causal,
            )
        )

        self.attention_weights = (
            attention_weights.detach()
        )

        # --------------------------------------------------
        # Transformer residual connection
        # --------------------------------------------------

        src = (
            src +
            self.dropout1(
                attention_output
            )
        )

        src = self.norm1(src)

        # --------------------------------------------------
        # Feed-forward network
        # --------------------------------------------------

        feedforward = self.linear1(src)

        feedforward = self.activation(
            feedforward
        )

        feedforward = self.dropout(
            feedforward
        )

        feedforward = self.linear2(
            feedforward
        )

        src = (
            src +
            self.dropout2(
                feedforward
            )
        )

        src = self.norm2(src)

        return src


# ==========================================================
# Load original trained model
# ==========================================================

print("=" * 70)
print("QUANTUM TRANSFORMER ATTENTION EXTRACTION")
print("=" * 70)

print(
    f"\nDevice : {DEVICE}"
)

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
    "✓ Original checkpoint loaded."
)


# ==========================================================
# Replace encoder layers with attention-enabled layers
# ==========================================================

attention_layers = nn.ModuleList()

for i in range(NUM_ENCODER_LAYERS):

    original_layer = (
        model.encoder.layers[i]
    )

    new_layer = AttentionEncoderLayer(
        d_model=D_MODEL,
        nhead=NHEAD,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Copy learned parameters
    new_layer.load_state_dict(
        original_layer.state_dict()
    )

    attention_layers.append(
        new_layer
    )

model.encoder.layers = attention_layers

print(
    f"✓ Replaced {NUM_ENCODER_LAYERS} "
    f"encoder layers with attention-enabled layers."
)

print(
    f"✓ Parameters preserved from checkpoint."
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

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

print(
    f"\nTest samples : {len(dataset)}"
)

print(
    f"Sequence     : {SEQUENCE_LENGTH}"
)

print(
    f"Attention heads : {NHEAD}"
)

print(
    f"Encoder layers  : {NUM_ENCODER_LAYERS}"
)


# ==========================================================
# First test sequence
# ==========================================================

sample_x, sample_y = next(
    iter(loader)
)

sample_x = sample_x.to(
    DEVICE
)

model.eval()

with torch.no_grad():

    prediction = model(
        sample_x
    )


print("\nInference")
print("=" * 70)

print(
    f"Prediction : "
    f"{prediction.item():.12e}"
)

print(
    f"Target     : "
    f"{sample_y.item():.12e}"
)


# ==========================================================
# Extract attention
# ==========================================================

layer_attention = []

print("\nAttention matrices")
print("=" * 70)

for i, layer in enumerate(
    model.encoder.layers
):

    weights = layer.attention_weights

    if weights is None:

        raise RuntimeError(
            f"No attention captured "
            f"from encoder layer {i}."
        )

    print(
        f"Layer {i + 1}: "
        f"{tuple(weights.shape)}"
    )

    # Shape:
    #
    # [batch, heads, query, key]
    #

    weights = weights.cpu()

    layer_attention.append(
        weights
    )


# ==========================================================
# Convert to numpy
# ==========================================================

layer_attention_np = [
    x.numpy()
    for x in layer_attention
]


# ==========================================================
# Save raw attention matrices
# ==========================================================

for i, attention in enumerate(
    layer_attention_np
):

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"layer_{i + 1}_attention.npy"
        ),
        attention
    )


# ==========================================================
# Average heads
# ==========================================================

layer_average_maps = []

for i, attention in enumerate(
    layer_attention_np
):

    # [batch, heads, query, key]
    #
    # Average batch and heads

    averaged = attention.mean(
        axis=(0, 1)
    )

    layer_average_maps.append(
        averaged
    )


# ==========================================================
# Individual layer heatmaps
# ==========================================================

for i, attention_map in enumerate(
    layer_average_maps
):

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        attention_map,
        aspect="auto",
        interpolation="nearest",
    )

    plt.xlabel(
        "Key Time Position"
    )

    plt.ylabel(
        "Query Time Position"
    )

    plt.title(
        f"Quantum Transformer "
        f"Attention — Layer {i + 1}"
    )

    plt.colorbar(
        label="Attention Weight"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"layer_{i + 1}_attention.png"
        ),
        dpi=300,
    )

    plt.close()


# ==========================================================
# Average across all layers
# ==========================================================

average_attention = np.mean(
    np.stack(
        layer_average_maps
    ),
    axis=0
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "average_attention.npy"
    ),
    average_attention
)


# ==========================================================
# Average attention heatmap
# ==========================================================

plt.figure(
    figsize=(9, 8)
)

plt.imshow(
    average_attention,
    aspect="auto",
    interpolation="nearest",
)

plt.xlabel(
    "Key Time Position"
)

plt.ylabel(
    "Query Time Position"
)

plt.title(
    "Quantum Transformer — "
    "Average Attention"
)

plt.colorbar(
    label="Attention Weight"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "average_attention.png"
    ),
    dpi=300,
)

plt.close()


# ==========================================================
# Temporal importance
# ==========================================================

temporal_importance = (
    average_attention.mean(
        axis=0
    )
)

importance_df = pd.DataFrame({

    "time_position":
        np.arange(
            SEQUENCE_LENGTH
        ),

    "attention_importance":
        temporal_importance,

})


importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "attention_temporal_importance.csv"
    ),
    index=False,
)


# ==========================================================
# Most important positions
# ==========================================================

top_indices = np.argsort(
    temporal_importance
)[::-1][:10]


print("\nMost attended time positions")
print("=" * 70)

for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank:02d}. "
        f"Position {index:03d} | "
        f"Attention "
        f"{temporal_importance[index]:.8f}"
    )


# ==========================================================
# Final
# ==========================================================

print("\n" + "=" * 70)
print("ATTENTION EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Layers captured : "
    f"{len(layer_attention_np)}"
)

print(
    f"Heads per layer : "
    f"{NHEAD}"
)

print(
    f"Attention shape : "
    f"{average_attention.shape}"
)

print(
    "\nSaved to:"
)

print(
    f"{OUTPUT_DIR}/"
)

print(
    "\n[PASS] Attention extraction completed."
)