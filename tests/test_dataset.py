from torch.utils.data import DataLoader

from ml.datasets.sequence_dataset import QuantumClockDataset

dataset = QuantumClockDataset(
    "data/processed/clock_train_v3_scaled.csv",
    sequence_length=128,
    target_column="true_detuning",
    time_column="time",
)

print("Dataset length:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
)

x, y = next(iter(loader))

print("Input shape :", x.shape)
print("Target shape:", y.shape)