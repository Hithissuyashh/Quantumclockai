import os
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
# CREATE CONTROLLER BRANCH
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

    # Same physical realization
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

    measured_offset = (
        discriminator.estimate_detuning(
            plus_probability,
            minus_probability,
        )
    )

    estimated_offset = (
        kalman.update(
            measured_offset
        )
    )

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
# FULL TELEMETRY FOR PRODUCTION AIPREDICTOR
# ==========================================================

def build_telemetry(
    result,
    env,
    noise,
):

    return {

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


# ==========================================================
# RUN ONE SEED
# ==========================================================

def run_seed(seed):

    print()
    print("=" * 70)
    print(f"RUNNING PRODUCTION VALIDATION SEED {seed}")
    print("=" * 70)

    # ------------------------------------------------------
    # Shared physical plant
    # ------------------------------------------------------

    environment = LaboratoryEnvironment(
        seed=seed
    )

    noise_model = NoiseModel(
        seed=seed
    )

    # ------------------------------------------------------
    # Independent branches
    # ------------------------------------------------------

    kalman_branch = create_branch()
    ai_branch = create_branch()

    # ------------------------------------------------------
    # Production AI
    # ------------------------------------------------------

    ai_predictor = AIPredictor()

    kalman_offsets = []
    ai_offsets = []

    ai_predictions = []
    ai_corrections = []

    # ------------------------------------------------------
    # Simulation
    # ------------------------------------------------------

    for step in range(STEPS):

        # Same environment realization
        env = environment.update()

        # Same physical noise realization
        noise = noise_model.total_noise(
            env
        )

        # --------------------------------------------------
        # Kalman-only branch
        # --------------------------------------------------

        kalman_result = controller_step(
            kalman_branch,
            env,
            noise
        )

        # --------------------------------------------------
        # AI branch
        # --------------------------------------------------

        ai_result = controller_step(
            ai_branch,
            env,
            noise
        )

        # --------------------------------------------------
        # Production telemetry
        # --------------------------------------------------

        telemetry = build_telemetry(
            ai_result,
            env,
            noise
        )

        # --------------------------------------------------
        # Production Transformer
        # --------------------------------------------------

        prediction = (
            ai_predictor.update(
                telemetry
            )
        )

        # --------------------------------------------------
        # AI correction
        # --------------------------------------------------

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

            ai_branch[
                "oscillator"
            ].apply_correction(
                ai_extra_correction
            )

        # --------------------------------------------------
        # Final AI offset
        # --------------------------------------------------

        ai_corrected_offset = (
            ai_branch[
                "oscillator"
            ].measure_error()
        )

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

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (
            (step + 1) % 2000 == 0
        ):

            print(
                f"{step + 1:5d}/{STEPS} | "
                f"Kalman: "
                f"{kalman_result['corrected_offset']:+.9e} | "
                f"AI: "
                f"{ai_corrected_offset:+.9e}"
            )

    # ======================================================
    # Exclude 128-sample warm-up
    # ======================================================

    start = SEQUENCE_LENGTH - 1

    kalman_values = np.asarray(
        kalman_offsets[start:],
        dtype=np.float64
    )

    ai_values = np.asarray(
        ai_offsets[start:],
        dtype=np.float64
    )

    prediction_values = np.asarray(
        ai_predictions[start:],
        dtype=np.float64
    )

    correction_values = np.asarray(
        ai_corrections[start:],
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

    prediction_count = int(
        np.sum(
            ~np.isnan(
                prediction_values
            )
        )
    )

    maximum_ai_correction = float(
        np.max(
            np.abs(
                correction_values
            )
        )
    )

    print()
    print("Seed results:")
    print(
        f"Kalman RMS : "
        f"{kalman_metrics['rms']:.12e}"
    )

    print(
        f"AI RMS     : "
        f"{ai_metrics['rms']:.12e}"
    )

    print(
        f"RMS improvement : "
        f"{rms_improvement:+.2f}%"
    )

    print(
        f"STD improvement : "
        f"{std_improvement:+.2f}%"
    )

    print(
        f"MAX improvement : "
        f"{max_improvement:+.2f}%"
    )

    print(
        f"AI predictions  : "
        f"{prediction_count}"
    )

    print(
        f"Max |AI correction| : "
        f"{maximum_ai_correction:.12e} Hz"
    )

    return {

        "seed":
            seed,

        "kalman_rms":
            kalman_metrics["rms"],

        "ai_rms":
            ai_metrics["rms"],

        "rms_improvement_percent":
            rms_improvement,

        "std_improvement_percent":
            std_improvement,

        "max_improvement_percent":
            max_improvement,

        "ai_prediction_count":
            prediction_count,

        "max_ai_correction":
            maximum_ai_correction,
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("PRODUCTION MULTI-SEED AI vs KALMAN VALIDATION")
    print("=" * 70)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device        : {device}")
    print(f"Seeds         : {SEEDS}")
    print(f"Steps / seed  : {STEPS}")
    print(f"AI gain       : {AI_GAIN}")
    print(
        f"AI correction : "
        f"±{AI_CORRECTION_LIMIT} Hz"
    )

    results = []

    for seed in SEEDS:

        result = run_seed(
            seed
        )

        results.append(
            result
        )

    # ======================================================
    # Aggregate
    # ======================================================

    results_df = pd.DataFrame(
        results
    )

    rms_mean = (
        results_df[
            "rms_improvement_percent"
        ].mean()
    )

    rms_std = (
        results_df[
            "rms_improvement_percent"
        ].std(
            ddof=1
        )
    )

    std_mean = (
        results_df[
            "std_improvement_percent"
        ].mean()
    )

    std_std = (
        results_df[
            "std_improvement_percent"
        ].std(
            ddof=1
        )
    )

    max_mean = (
        results_df[
            "max_improvement_percent"
        ].mean()
    )

    max_std = (
        results_df[
            "max_improvement_percent"
        ].std(
            ddof=1
        )
    )

    # ======================================================
    # Display
    # ======================================================

    print()
    print("=" * 70)
    print("PRODUCTION MULTI-SEED VALIDATION RESULTS")
    print("=" * 70)

    print()
    print(
        "Per-seed RMS improvement:"
    )

    for _, row in results_df.iterrows():

        print(
            f"Seed {int(row['seed']):4d} : "
            f"{row['rms_improvement_percent']:+.2f}%"
        )

    print()
    print(
        "Per-seed STD improvement:"
    )

    for _, row in results_df.iterrows():

        print(
            f"Seed {int(row['seed']):4d} : "
            f"{row['std_improvement_percent']:+.2f}%"
        )

    print()
    print(
        "Per-seed maximum-offset improvement:"
    )

    for _, row in results_df.iterrows():

        print(
            f"Seed {int(row['seed']):4d} : "
            f"{row['max_improvement_percent']:+.2f}%"
        )

    print()
    print("-" * 70)
    print("AGGREGATE")
    print("-" * 70)

    print(
        f"RMS improvement : "
        f"{rms_mean:+.2f}% ± {rms_std:.2f}%"
    )

    print(
        f"STD improvement : "
        f"{std_mean:+.2f}% ± {std_std:.2f}%"
    )

    print(
        f"MAX improvement : "
        f"{max_mean:+.2f}% ± {max_std:.2f}%"
    )

    # ======================================================
    # Production sanity checks
    # ======================================================

    total_expected = (
        len(SEEDS)
        *
        (STEPS - (SEQUENCE_LENGTH - 1))
    )

    total_predictions = int(
        results_df[
            "ai_prediction_count"
        ].sum()
    )

    print()
    print(
        f"AI predictions : "
        f"{total_predictions}/{total_expected}"
    )

    print(
        f"Maximum observed AI correction : "
        f"{results_df['max_ai_correction'].max():.12e} Hz"
    )

    # ======================================================
    # Save
    # ======================================================

    output_dir = (
        "results/"
        "live_ai_vs_kalman_multiseed"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    results_df.to_csv(
        f"{output_dir}/production_multiseed_results.csv",
        index=False
    )

    pd.DataFrame({

        "metric": [
            "rms_mean",
            "rms_std",
            "std_mean",
            "std_std",
            "max_mean",
            "max_std",
        ],

        "value": [
            rms_mean,
            rms_std,
            std_mean,
            std_std,
            max_mean,
            max_std,
        ],

    }).to_csv(
        f"{output_dir}/production_multiseed_summary.csv",
        index=False
    )

    print()
    print("Saved:")
    print(
        f"{output_dir}/production_multiseed_results.csv"
    )
    print(
        f"{output_dir}/production_multiseed_summary.csv"
    )

    print()
    print(
        "[PASS] Production multi-seed "
        "validation completed."
    )


if __name__ == "__main__":
    main()