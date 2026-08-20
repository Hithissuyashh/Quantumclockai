import torch
import torch.nn as nn

from ml.config import (
    NUM_FEATURES,
    D_MODEL,
    NHEAD,
    NUM_ENCODER_LAYERS,
    DIM_FEEDFORWARD,
    DROPOUT,
)

from ml.models.positional_encoding import PositionalEncoding


class QuantumTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        # Project 20 features into embedding space
        self.input_projection = nn.Linear(
            NUM_FEATURES,
            D_MODEL
        )

        self.position = PositionalEncoding(
            D_MODEL,
            DROPOUT
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_ENCODER_LAYERS
        )

        # Regression head
        self.head = nn.Sequential(

            nn.Linear(D_MODEL, 64),

            nn.GELU(),

            nn.Dropout(DROPOUT),

            nn.Linear(64, 1)

        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.position(x)

        x = self.encoder(x)

        # Global Average Pooling
        x = x.mean(dim=1)

        y = self.head(x)

        return y.squeeze(-1)
