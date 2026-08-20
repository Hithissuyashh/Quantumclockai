import torch

from ml.training.metrics import (
    mse,
    mae,
    rmse,
)

prediction = torch.tensor(
    [1.0, 2.0, 3.0]
)

target = torch.tensor(
    [1.1, 2.2, 2.9]
)

print("MSE :", mse(prediction, target))
print("MAE :", mae(prediction, target))
print("RMSE:", rmse(prediction, target))
