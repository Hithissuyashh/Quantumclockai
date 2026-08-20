import matplotlib.pyplot as plt
import torch


def prediction_plot(predictions, targets):

    plt.figure(figsize=(10,5))

    plt.plot(
        targets[:500],
        label="Ground Truth",
        linewidth=2
    )

    plt.plot(
        predictions[:500],
        label="Prediction",
        linewidth=1.5
    )

    plt.xlabel("Sample")

    plt.ylabel("Fractional Frequency")

    plt.title("Prediction vs Ground Truth")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("results/prediction_vs_truth.png", dpi=300)

    plt.show()


def residual_plot(predictions, targets):

    residual = predictions - targets

    plt.figure(figsize=(10,5))

    plt.plot(residual[:500])

    plt.title("Residual Error")

    plt.xlabel("Sample")

    plt.ylabel("Prediction Error")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("results/residuals.png", dpi=300)

    plt.show()


def histogram(predictions, targets):

    residual = predictions - targets

    plt.figure(figsize=(8,5))

    plt.hist(
        residual,
        bins=50
    )

    plt.title("Residual Distribution")

    plt.xlabel("Residual")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig("results/error_histogram.png", dpi=300)

    plt.show()