import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class QuantumClockDataset(Dataset):
    """
    Sliding-window dataset for chronological
    quantum-clock forecasting.
    """

    def __init__(
        self,
        csv_path,
        sequence_length=128,
        target_column="true_detuning",
        time_column="time",
    ):

        self.df = pd.read_csv(csv_path)

        self.sequence_length = sequence_length

        self.features = self.df.drop(
            columns=[
                target_column,
                time_column,
            ]
        ).values.astype(np.float32)

        self.targets = (
            self.df[target_column]
            .values
            .astype(np.float32)
        )

    def __len__(self):

        return max(
            0,
            len(self.features) -
            self.sequence_length
        )

    def __getitem__(self, idx):

        x = self.features[
            idx:
            idx + self.sequence_length
        ]

        y = self.targets[
            idx + self.sequence_length
        ]

        return (
            torch.tensor(x),
            torch.tensor(y),
        )