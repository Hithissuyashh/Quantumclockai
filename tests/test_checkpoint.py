import torch

from ml.models.transformer_model import QuantumTransformer
from ml.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

model = QuantumTransformer()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

save_checkpoint(
    model,
    optimizer,
    epoch=5,
    loss=0.123,
    path="temp_checkpoint.pth",
)

epoch, loss = load_checkpoint(
    model,
    optimizer,
    "temp_checkpoint.pth",
    "cpu",
)

print("Epoch :", epoch)
print("Loss  :", loss)