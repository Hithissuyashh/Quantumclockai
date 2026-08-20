import torch
from tqdm import tqdm

from ml.training.metrics import (
    mse,
    mae,
    rmse,
)


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        scheduler=None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=device.type == "cuda"
        )

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0

        progress = tqdm(
            loader,
            desc="Training",
            leave=False
        )

        for x, y in progress:

            x = x.to(
                self.device,
                non_blocking=True
            )

            y = y.to(
                self.device,
                non_blocking=True
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=self.device.type == "cuda"
            ):

                prediction = self.model(x)

                loss = self.criterion(
                    prediction,
                    y
                )

            self.scaler.scale(loss).backward()

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            total_loss += loss.item()

            total_mae += mae(
                prediction.detach(),
                y
            )

            total_rmse += rmse(
                prediction.detach(),
                y
            )

            progress.set_postfix(
                loss=f"{loss.item():.6f}"
            )

        if self.scheduler is not None:
            self.scheduler.step()

        n = len(loader)

        return {

            "loss": total_loss / n,

            "mae": total_mae / n,

            "rmse": total_rmse / n,

        }

    @torch.no_grad()
    def validate(self, loader):

        self.model.eval()

        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0

        progress = tqdm(
            loader,
            desc="Validation",
            leave=False
        )

        for x, y in progress:

            x = x.to(
                self.device,
                non_blocking=True
            )

            y = y.to(
                self.device,
                non_blocking=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=self.device.type == "cuda"
            ):

                prediction = self.model(x)

                loss = self.criterion(
                    prediction,
                    y
                )

            total_loss += loss.item()

            total_mae += mae(
                prediction,
                y
            )

            total_rmse += rmse(
                prediction,
                y
            )

        n = len(loader)

        return {

            "loss": total_loss / n,

            "mae": total_mae / n,

            "rmse": total_rmse / n,

        }