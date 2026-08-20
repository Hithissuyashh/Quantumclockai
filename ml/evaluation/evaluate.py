import torch

from torch.utils.data import DataLoader

from ml.config import CHECKPOINT_DIR, BATCH_SIZE
from ml.datasets.sequence_dataset import QuantumClockDataset
from ml.models.transformer_model import QuantumTransformer
from ml.training.metrics import mse, mae, rmse
from ml.evaluation.plots import (
    prediction_plot,
    residual_plot,
    histogram,
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using Device : {device}")

    # ------------------------------------------------
    # Test dataset
    # ------------------------------------------------

    test_dataset = QuantumClockDataset(
        "data/processed/clock_test_v3_scaled.csv",
        sequence_length=128,
        target_column="true_detuning",
        time_column="time",
    )

    print(f"Test samples : {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------
    # Load model
    # ------------------------------------------------

    model = QuantumTransformer().to(device)

    checkpoint = torch.load(
        CHECKPOINT_DIR / "best_model.pth",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ------------------------------------------------
    # Inference
    # ------------------------------------------------

    predictions = []
    targets = []

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)

            prediction = model(x)

            predictions.append(
                prediction.cpu()
            )

            targets.append(y)

    predictions = torch.cat(predictions)

    targets = torch.cat(targets)

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------

    print("\nEvaluation Results")
    print("=" * 40)

    print("MSE  :", mse(predictions, targets))
    print("MAE  :", mae(predictions, targets))
    print("RMSE :", rmse(predictions, targets))

    prediction_plot(
        predictions.numpy(),
        targets.numpy(),
    )

    residual_plot(
        predictions.numpy(),
        targets.numpy(),
    )

    histogram(
        predictions.numpy(),
        targets.numpy(),
    )


if __name__ == "__main__":
    main()