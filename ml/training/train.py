import torch

from torch.utils.data import DataLoader

from ml.config import *

from ml.datasets.sequence_dataset import QuantumClockDataset
from ml.models.transformer_model import QuantumTransformer

from ml.training.trainer import Trainer
from ml.training.checkpoint import save_checkpoint
from ml.training.utils import get_device, set_seed


def main():

    # ------------------------------------------------
    # Reproducibility
    # ------------------------------------------------

    set_seed(RANDOM_SEED)

    device = get_device()

    print(f"\nUsing Device : {device}\n")

    # ------------------------------------------------
    # Chronological datasets
    # ------------------------------------------------

    train_dataset = QuantumClockDataset(
        "data/processed/clock_train_v3_scaled.csv",
        sequence_length=128,
        target_column="true_detuning",
        time_column="time",
    )

    val_dataset = QuantumClockDataset(
        "data/processed/clock_val_v3_scaled.csv",
        sequence_length=128,
        target_column="true_detuning",
        time_column="time",
    )

    test_dataset = QuantumClockDataset(
        "data/processed/clock_test_v3_scaled.csv",
        sequence_length=128,
        target_column="true_detuning",
        time_column="time",
    )

    print("Dataset sizes:")
    print("Train :", len(train_dataset))
    print("Val   :", len(val_dataset))
    print("Test  :", len(test_dataset))

    # ------------------------------------------------
    # DataLoaders
    # ------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    model = QuantumTransformer().to(device)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        scheduler,
    )

    best_loss = float("inf")

    print("\n" + "=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    for epoch in range(EPOCHS):

        train_metrics = trainer.train_epoch(
            train_loader
        )

        val_metrics = trainer.validate(
            val_loader
        )

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train Loss {train_metrics['loss']:.8f} | "
            f"Val Loss {val_metrics['loss']:.8f} | "
            f"Val RMSE {val_metrics['rmse']:.8f}"
        )

        if val_metrics["loss"] < best_loss:

            best_loss = val_metrics["loss"]

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_loss,
                CHECKPOINT_DIR / "best_model.pth",
            )

            print("✓ Best model saved.")

    print("\nTraining Finished.")
    print(
        f"Best Validation Loss : "
        f"{best_loss:.8f}"
    )


if __name__ == "__main__":
    main()