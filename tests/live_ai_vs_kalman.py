import numpy as np
import pandas as pd
import torch

from simulator.environment import LaboratoryEnvironment
from simulator.noise import NoiseModel
from simulator.oscillator import OpticalOscillator
from simulator.atomic_reference import AtomicReference
from simulator.ramsey import RamseyInterrogation
from simulator.discriminator import FrequencyDiscriminator

from controller.servo import ServoController
from estimator.kalman import KalmanFilter

from ml.inference.ai_predictor import AIPredictor


# ==========================================================
# CONFIGURATION
# ==========================================================

SEEDS = [123, 456, 789, 1000]
STEPS = 10000
SEQUENCE_LENGTH = 128

AI_GAIN = 0.05
AI_CORRECTION_LIMIT = 0.0002


# ==========================================================
# METRICS
# ==========================================================

def metrics(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "rms": np.sqrt(
            np.mean(values ** 2)
        ),
        "max_abs": np.max(
            np.abs(values)
        ),
    }


# ==========================================================
# CREATE CLASSICAL CONTROLLER BRANCH
# ==========================================================

def create_branch():

    return {
        "oscillator":
            OpticalOscillator(),

        "atomic_reference":
            AtomicReference(),

        "ramsey":
            RamseyInterrogation(
                interrogation_time=0.1
            ),

        "discriminator":
            FrequencyDiscriminator(
                probe_offset=2.5
            ),

        "kalman":
            KalmanFilter(),

        "servo":
            ServoController(),
    }


# ==========================================================
# CLASSICAL CONTROLLER STEP
# ==========================================================

def controller_step(
    branch,
    env,
    noise,
):

    oscillator = branch["oscillator"]

    atomic_reference = (
        branch["atomic_reference"]
    )

    ramsey = branch["ramsey"]

    discriminator = (
        branch["discriminator"]
    )

    kalman = branch["kalman"]

    servo = branch["servo"]

    # ------------------------------------------------------
    # SAME physical noise
    # ------------------------------------------------------

    oscillator.apply_noise(
        noise["total"]
    )

    measured_frequency = (
        oscillator.frequency()
    )

    atomic_frequency = (
        atomic_reference.frequency()
    )

    true_detuning = (
        oscillator.measure_error()
    )

    # ------------------------------------------------------
    # Ramsey interrogation
    # ------------------------------------------------------

    plus_probability = (
        ramsey.excitation_probability(
            true_detuning
            + discriminator.probe_offset
        )
    )

    minus_probability = (
        ramsey.excitation_probability(
            true_detuning
            - discriminator.probe_offset
        )
    )

    # ------------------------------------------------------
    # Frequency discriminator
    # ------------------------------------------------------

    measured_offset = (
        discriminator.estimate_detuning(
            plus_probability,
            minus_probability,
        )
    )

    # ------------------------------------------------------
    # Kalman estimation
    # ------------------------------------------------------

    estimated_offset = (
        kalman.update(
            measured_offset
        )
    )

    # ------------------------------------------------------
    # PI servo
    # ------------------------------------------------------

    servo_correction = (
        servo.update(
            estimated_offset
        )
    )

    oscillator.apply_correction(
        servo_correction
    )

    corrected_offset = (
        oscillator.measure_error()
    )

    return {

        "time":
            env["time"],

        "true_detuning":
            true_detuning,

        "measured_frequency":
            measured_frequency,

        "atomic_frequency":
            atomic_frequency,

        "measured_offset":
            measured_offset,

        "estimated_offset":
            estimated_offset,

        "servo_correction":
            servo_correction,

        "excitation_probability_plus":
            plus_probability,

        "excitation_probability_minus":
            minus_probability,

        "corrected_offset":
            corrected_offset,
    }


# ==========================================================
# BUILD 20-FEATURE TELEMETRY
#
# Exact order used by feature_scaler_v3.pkl
# ==========================================================

def build_telemetry(
    result,
    env,
    noise,
):

    return {

        # --------------------------------------------------
        # Full simulator telemetry expected by AIPredictor
        # --------------------------------------------------

        "time":
            env["time"],

        "environment":
            env,

        "noise":
            noise,

        "atomic_frequency":
            result["atomic_frequency"],

        "true_detuning":
            result["true_detuning"],

        "measured_frequency":
            result["measured_frequency"],

        "measured_offset":
            result["measured_offset"],

        "estimated_offset":
            result["estimated_offset"],

        "servo_correction":
            result["servo_correction"],

        "excitation_probability_plus":
            result[
                "excitation_probability_plus"
            ],

        "excitation_probability_minus":
            result[
                "excitation_probability_minus"
            ],

        "corrected_offset":
            result["corrected_offset"],
    }

    return {

        "temperature":
            env["temperature"],

        "magnetic":
            env["magnetic"],

        "laser_power":
            env["laser_power"],

        "pressure":
            env["pressure"],

        "humidity":
            env["humidity"],

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
            result["measured_offset"],

        "estimated_offset":
            result["estimated_offset"],

        "servo_correction":
            result["servo_correction"],

        "excitation_probability_plus":
            result[
                "excitation_probability_plus"
            ],

        "excitation_probability_minus":
            result[
                "excitation_probability_minus"
            ],
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("LIVE AI vs KALMAN-ONLY BENCHMARK")
    print("=" * 70)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device        : {device}")
    print(f"Seed          : {SEED}")
    print(f"Steps         : {STEPS}")
    print(f"AI gain       : {AI_GAIN}")
    print(
        f"AI correction : "
        f"±{AI_CORRECTION_LIMIT} Hz"
    )
    print(
        f"Sequence      : "
        f"{SEQUENCE_LENGTH}"
    )

    # ------------------------------------------------------
    # Shared physical plant
    # ------------------------------------------------------

    print()
    print(
        "✓ Creating shared laboratory "
        "environment..."
    )

    environment = LaboratoryEnvironment(
        seed=SEED
    )

    noise_model = NoiseModel(
        seed=SEED
    )

    # ------------------------------------------------------
    # Independent branches
    # ------------------------------------------------------

    kalman_branch = create_branch()
    ai_branch = create_branch()

    # ------------------------------------------------------
    # Production AIPredictor
    # ------------------------------------------------------

    print(
        "✓ Creating production "
        "AIPredictor..."
    )

    ai_predictor = AIPredictor()

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Both branches receive exactly "
        "the same environment and "
        "physical noise."
    )
    print(
        "The AI branch uses the "
        "production AIPredictor."
    )

    # ------------------------------------------------------
    # Storage
    # ------------------------------------------------------

    ai_history = []

    kalman_offsets = []
    ai_offsets = []

    ai_predictions = []
    ai_corrections = []

    # ------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING LIVE SHARED-PLANT EXPERIMENT")
    print("=" * 70)

    for step in range(STEPS):

        # ==================================================
        # ONE environment realization
        # ==================================================

        env = environment.update()

        # ==================================================
        # ONE physical-noise realization
        # ==================================================

        noise = noise_model.total_noise(
            env
        )

        # ==================================================
        # KALMAN-ONLY BRANCH
        # ==================================================

        kalman_result = controller_step(
            kalman_branch,
            env,
            noise
        )

        # ==================================================
        # AI BRANCH
        #
        # First execute the same classical
        # Kalman + PI controller.
        # ==================================================

        ai_result = controller_step(
            ai_branch,
            env,
            noise
        )

        # ==================================================
        # Build exact 20-feature telemetry
        # ==================================================

        telemetry = build_telemetry(
            ai_result,
            env,
            noise
        )

        # ==================================================
        # Production AI inference
        #
        # AIPredictor internally:
        #   - buffers 128 samples
        #   - scales using v3 scaler
        #   - runs Transformer
        # ==================================================

        prediction = (
            ai_predictor.update(
                telemetry
            )
        )

        # ==================================================
        # AI extra correction
        # ==================================================

        ai_extra_correction = 0.0

        if prediction is not None:

            ai_extra_correction = (
                AI_GAIN * prediction
            )

            ai_extra_correction = float(
                np.clip(
                    ai_extra_correction,
                    -AI_CORRECTION_LIMIT,
                    AI_CORRECTION_LIMIT
                )
            )

            # Apply AI correction AFTER
            # the classical PI correction,
            # exactly like the validated benchmark.

            ai_branch[
                "oscillator"
            ].apply_correction(
                ai_extra_correction
            )

        # ==================================================
        # Final AI corrected offset
        # ==================================================

        ai_corrected_offset = (
            ai_branch[
                "oscillator"
            ].measure_error()
        )

        # ==================================================
        # Record
        # ==================================================

        kalman_offsets.append(
            kalman_result[
                "corrected_offset"
            ]
        )

        ai_offsets.append(
            ai_corrected_offset
        )

        ai_predictions.append(
            np.nan
            if prediction is None
            else prediction
        )

        ai_corrections.append(
            ai_extra_correction
        )

        # ==================================================
        # Progress
        # ==================================================

        if (
            (step + 1) % 1000 == 0
        ):

            print(
                f"{step + 1:5d}/{STEPS} | "
                f"Kalman: "
                f"{kalman_result['corrected_offset']:+.9e} | "
                f"AI: "
                f"{ai_corrected_offset:+.9e}"
            )

    # ======================================================
    # Remove Transformer warm-up
    # ======================================================

    kalman_values = np.asarray(
        kalman_offsets[
            SEQUENCE_LENGTH - 1:
        ],
        dtype=np.float64
    )

    ai_values = np.asarray(
        ai_offsets[
            SEQUENCE_LENGTH - 1:
        ],
        dtype=np.float64
    )

    prediction_values = np.asarray(
        ai_predictions[
            SEQUENCE_LENGTH - 1:
        ],
        dtype=np.float64
    )

    correction_values = np.asarray(
        ai_corrections[
            SEQUENCE_LENGTH - 1:
        ],
        dtype=np.float64
    )

    # ======================================================
    # Metrics
    # ======================================================

    kalman_metrics = metrics(
        kalman_values
    )

    ai_metrics = metrics(
        ai_values
    )

    rms_improvement = (
        1.0
        -
        ai_metrics["rms"]
        /
        kalman_metrics["rms"]
    ) * 100.0

    std_improvement = (
        1.0
        -
        ai_metrics["std"]
        /
        kalman_metrics["std"]
    ) * 100.0

    max_improvement = (
        1.0
        -
        ai_metrics["max_abs"]
        /
        kalman_metrics["max_abs"]
    ) * 100.0

    # ======================================================
    # Results
    # ======================================================

    print()
    print("=" * 70)
    print("LIVE AI vs KALMAN RESULTS")
    print("=" * 70)

    print()
    print(
        f"Evaluation samples : "
        f"{len(kalman_values)}"
    )

    print(
        f"Warm-up excluded   : "
        f"{SEQUENCE_LENGTH - 1}"
    )

    print()
    print("KALMAN-ONLY")

    print(
        f"Mean offset : "
        f"{kalman_metrics['mean']:+.12e} Hz"
    )

    print(
        f"STD         : "
        f"{kalman_metrics['std']:.12e} Hz"
    )

    print(
        f"RMS         : "
        f"{kalman_metrics['rms']:.12e} Hz"
    )

    print(
        f"Maximum     : "
        f"{kalman_metrics['max_abs']:.12e} Hz"
    )

    print()
    print("AI-ASSISTED")

    print(
        f"Mean offset : "
        f"{ai_metrics['mean']:+.12e} Hz"
    )

    print(
        f"STD         : "
        f"{ai_metrics['std']:.12e} Hz"
    )

    print(
        f"RMS         : "
        f"{ai_metrics['rms']:.12e} Hz"
    )

    print(
        f"Maximum     : "
        f"{ai_metrics['max_abs']:.12e} Hz"
    )

    print()
    print("IMPROVEMENT")

    print(
        f"RMS improvement : "
        f"{rms_improvement:+.2f}%"
    )

    print(
        f"STD improvement : "
        f"{std_improvement:+.2f}%"
    )

    print(
        f"Maximum improvement : "
        f"{max_improvement:+.2f}%"
    )

    print()
    print(
        "AI prediction count : "
        f"{np.sum(~np.isnan(prediction_values))}"
    )

    print(
        "Maximum |AI correction| : "
        f"{np.max(np.abs(correction_values)):.12e} Hz"
    )

    # ======================================================
    # Save results separately
    # ======================================================

    output_dir = (
        "results/live_ai_vs_kalman"
    )

    import os

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    pd.DataFrame({

        "step":
            np.arange(
                SEQUENCE_LENGTH,
                STEPS + 1
            ),

        "kalman_corrected_offset":
            kalman_values,

        "ai_corrected_offset":
            ai_values,

        "ai_prediction":
            prediction_values,

        "ai_extra_correction":
            correction_values,

    }).to_csv(
        f"{output_dir}/seed_42_results.csv",
        index=False
    )

    pd.DataFrame({

        "metric": [
            "kalman_mean",
            "kalman_std",
            "kalman_rms",
            "kalman_max_abs",
            "ai_mean",
            "ai_std",
            "ai_rms",
            "ai_max_abs",
            "rms_improvement_percent",
            "std_improvement_percent",
            "max_improvement_percent",
        ],

        "value": [
            kalman_metrics["mean"],
            kalman_metrics["std"],
            kalman_metrics["rms"],
            kalman_metrics["max_abs"],
            ai_metrics["mean"],
            ai_metrics["std"],
            ai_metrics["rms"],
            ai_metrics["max_abs"],
            rms_improvement,
            std_improvement,
            max_improvement,
        ],

    }).to_csv(
        f"{output_dir}/seed_42_summary.csv",
        index=False
    )

    print()
    print("Saved:")
    print(
        f"{output_dir}/seed_42_results.csv"
    )
    print(
        f"{output_dir}/seed_42_summary.csv"
    )

    print()
    print(
        "[PASS] Live Seed 42 benchmark completed."
    )


if __name__ == "__main__":
    main()