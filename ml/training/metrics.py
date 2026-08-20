import torch


def mse(prediction, target):
    return torch.mean((prediction - target) ** 2).item()


def mae(prediction, target):
    return torch.mean(torch.abs(prediction - target)).item()


def rmse(prediction, target):
    return torch.sqrt(
        torch.mean(
            (prediction - target) ** 2
        )
    ).item()
