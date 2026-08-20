import torch

from ml.models.transformer_model import QuantumTransformer

model = QuantumTransformer()

x = torch.randn(
    32,
    128,
    20
)

y = model(x)

print("Input :", x.shape)

print("Output:", y.shape)

print()

total = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Parameters : {total:,}")
