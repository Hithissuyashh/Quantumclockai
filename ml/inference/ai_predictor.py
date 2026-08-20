from collections import deque
from pathlib import Path

import joblib
import numpy as np
import torch

from ml.models.transformer_model import QuantumTransformer


class AIPredictor:
    """
    Live inference adapter for the trained QuantumClock Transformer.

    Pipeline:

        raw 20 features
            ↓
        feature_scaler_v3
            ↓
        128-step sequence
            ↓
        QuantumTransformer
            ↓
        predicted true_detuning

    The predictor does NOT modify the clock physics.
    """

    FEATURE_NAMES = [
        "temperature",
        "magnetic",
        "laser_power",
        "pressure",
        "humidity",
        "white",
        "random_walk",
        "flicker",
        "laser_phase",
        "qpn",
        "temperature_shift",
        "zeeman",
        "blackbody",
        "aging",
        "total_noise",
        "measured_offset",
        "estimated_offset",
        "servo_correction",
        "excitation_probability_plus",
        "excitation_probability_minus",
    ]

    SEQUENCE_LENGTH = 128

    def __init__(
        self,
        checkpoint_path="checkpoints/best_model.pth",
        scaler_path="data/processed/feature_scaler_v3.pkl",
        device=None,
    ):

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        # --------------------------------------------------
        # Paths
        # --------------------------------------------------

        self.checkpoint_path = Path(checkpoint_path)
        self.scaler_path = Path(scaler_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Transformer checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Feature scaler not found: "
                f"{self.scaler_path}"
            )

        # --------------------------------------------------
        # Scaler
        # --------------------------------------------------

        self.scaler = joblib.load(
            self.scaler_path
        )

        # --------------------------------------------------
        # Validate scaler feature contract
        # --------------------------------------------------

        scaler_features = list(
            self.scaler.feature_names_in_
        )

        if scaler_features != self.FEATURE_NAMES:

            raise ValueError(
                "Scaler feature order does not match "
                "the trained Transformer feature contract.\n\n"
                f"Expected:\n{self.FEATURE_NAMES}\n\n"
                f"Scaler:\n{scaler_features}"
            )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        self.model = QuantumTransformer().to(
            self.device
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # --------------------------------------------------
        # Rolling feature history
        # --------------------------------------------------

        self.history = deque(
            maxlen=self.SEQUENCE_LENGTH
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        self.prediction_count = 0

        print("✓ AIPredictor initialized.")
        print(f"Device          : {self.device}")
        print(f"Features        : {len(self.FEATURE_NAMES)}")
        print(f"Sequence length : {self.SEQUENCE_LENGTH}")
        print(f"Model           : {self.checkpoint_path}")
        print(f"Scaler          : {self.scaler_path}")

    # ======================================================
    # Feature construction
    # ======================================================

    def build_feature_vector(self, telemetry):
        """
        Extract the exact 20 raw features required by the
        trained Transformer.

        `telemetry` is expected to contain:

            environment
            noise
            measured_offset
            estimated_offset
            servo_correction
            excitation_probability_plus
            excitation_probability_minus
        """

        environment = telemetry["environment"]
        noise = telemetry["noise"]

        features = {

            "temperature":
                environment["temperature"],

            "magnetic":
                environment["magnetic"],

            "laser_power":
                environment["laser_power"],

            "pressure":
                environment["pressure"],

            "humidity":
                environment["humidity"],

            "white":
                noise["white"],

            "random_walk":
                noise["random_walk"],

            "flicker":
                noise["flicker"],

            "laser_phase":
                noise["laser_phase"],

            "qpn":
                noise["qpn"],

            "temperature_shift":
                noise["temperature"],

            "zeeman":
                noise["zeeman"],

            "blackbody":
                noise["blackbody"],

            "aging":
                noise["aging"],

            "total_noise":
                noise["total"],

            "measured_offset":
                telemetry["measured_offset"],

            "estimated_offset":
                telemetry["estimated_offset"],

            "servo_correction":
                telemetry["servo_correction"],

            "excitation_probability_plus":
                telemetry["excitation_probability_plus"],

            "excitation_probability_minus":
                telemetry["excitation_probability_minus"],
        }

        return np.asarray(
            [
                features[name]
                for name in self.FEATURE_NAMES
            ],
            dtype=np.float32,
        )

    # ======================================================
    # Add one timestep
    # ======================================================

    def update(self, telemetry):
        """
        Add one raw telemetry sample to the rolling
        128-step inference buffer.

        Returns:

            None

        until the sequence is full.
        """

        feature_vector = (
            self.build_feature_vector(
                telemetry
            )
        )

        if feature_vector.shape != (20,):
            raise RuntimeError(
                "Invalid AI feature vector shape: "
                f"{feature_vector.shape}"
            )

        self.history.append(
            feature_vector
        )

        if len(self.history) < self.SEQUENCE_LENGTH:
            return None

        return self.predict()

    # ======================================================
    # Prediction
    # ======================================================

    @torch.no_grad()
    def predict(self):
        """
        Run Transformer inference using the current
        128-step rolling history.
        """

        if len(self.history) < self.SEQUENCE_LENGTH:

            return None

        raw_sequence = np.asarray(
            self.history,
            dtype=np.float32,
        )

        if raw_sequence.shape != (
            self.SEQUENCE_LENGTH,
            len(self.FEATURE_NAMES),
        ):

            raise RuntimeError(
                "Invalid sequence shape: "
                f"{raw_sequence.shape}"
            )

        # --------------------------------------------------
        # Scale using the SAME scaler used during training
        # --------------------------------------------------

        import pandas as pd

        scaled_sequence = self.scaler.transform(
            pd.DataFrame(
                raw_sequence,
                columns=self.FEATURE_NAMES,
            )
        )

        # --------------------------------------------------
        # Tensor
        # --------------------------------------------------

        x = torch.from_numpy(
            scaled_sequence
        ).unsqueeze(0).to(
            self.device,
            dtype=torch.float32,
        )

        # Shape:
        # [1, 128, 20]

        prediction = self.model(x)

        prediction = (
            prediction.squeeze()
            .detach()
            .cpu()
            .item()
        )

        self.prediction_count += 1

        return float(prediction)

    # ======================================================
    # Status
    # ======================================================

    @property
    def ready(self):
        """
        True once the 128-step history is full.
        """

        return (
            len(self.history)
            >= self.SEQUENCE_LENGTH
        )

    @property
    def history_length(self):
        return len(self.history)

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):
        """
        Clear the live inference history.
        """

        self.history.clear()

        self.prediction_count = 0

    # ======================================================
    # Representation
    # ======================================================

    def __repr__(self):

        return (
            f"<AIPredictor "
            f"device={self.device}, "
            f"features={len(self.FEATURE_NAMES)}, "
            f"sequence={self.SEQUENCE_LENGTH}, "
            f"ready={self.ready}>"
        )