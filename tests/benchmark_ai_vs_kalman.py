import os
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from simulator.environment import LaboratoryEnvironment
from simulator.noise import NoiseModel
from simulator.oscillator import OpticalOscillator
from simulator.atomic_reference import AtomicReference
from simulator.ramsey import RamseyInterrogation
from simulator.discriminator import FrequencyDiscriminator

from controller.servo import ServoController
from estimator.kalman import KalmanFilter

from ml.models.transformer_model import QuantumTransformer
from ml.config import CHECKPOINT_DIR


# ==========================================================
# CONFIGURATION
# ==========================================================

SEEDS = [42, 123, 456, 789, 1000]

STEPS = 10_000
SEQUENCE_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = (
    CHECKPOINT_DIR / "best_model.pth"
)

SCALER_PATH = (
    "data/processed/feature_scaler_v3.pkl"
)

OUTPUT_DIR = (
    "results/ai_vs_kalman_multiseed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# AI CONTROL
# ==========================================================

AI_GAIN = 0.05

AI_CORRECTION_LIMIT = 0.0002


# ==========================================================
# EXACT 20 FEATURES
# ==========================================================

FEATURE_COLUMNS = [

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

assert len(FEATURE_COLUMNS) == 20


# ==========================================================
# HEADER
# ==========================================================

print("=" * 70)
print("MULTI-SEED AI vs KALMAN-ONLY BENCHMARK")
print("=" * 70)

print(
    f"\nDevice        : {DEVICE}"
)

print(
    f"Seeds         : {SEEDS}"
)

print(
    f"Steps / seed  : {STEPS}"
)

print(
    f"AI gain       : {AI_GAIN}"
)

print(
    f"AI correction : ±{AI_CORRECTION_LIMIT} Hz"
)


# ==========================================================
# LOAD SCALER
# ==========================================================

scaler = joblib.load(
    SCALER_PATH
)

print(
    "\n✓ v3 scaler loaded."
)

if hasattr(
    scaler,
    "n_features_in_"
):

    print(
        f"Scaler features : "
        f"{scaler.n_features_in_}"
    )

    if scaler.n_features_in_ != 20:

        raise RuntimeError(
            "Scaler must contain exactly 20 features."
        )


# ==========================================================
# LOAD TRANSFORMER
# ==========================================================

model = QuantumTransformer().to(
    DEVICE
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "✓ Transformer loaded."
)


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
# CONTROLLER STEP
# ==========================================================

def controller_step(
    branch,
    env,
    noise
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


    # Same physical noise
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
            true_detuning +
            discriminator.probe_offset
        )
    )

    minus_probability = (
        ramsey.excitation_probability(
            true_detuning -
            discriminator.probe_offset
        )
    )


    measured_offset = (
        discriminator.estimate_detuning(
            plus_probability,
            minus_probability
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

        "time": env["time"],

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
# FEATURE VECTOR
# ==========================================================

def build_features(
    result,
    env,
    noise
):

    values = {

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

    return np.asarray(
        [
            values[column]
            for column in FEATURE_COLUMNS
        ],
        dtype=np.float64
    )


# ==========================================================
# TRANSFORMER
# ==========================================================

@torch.no_grad()
def transformer_predict(
    sequence
):

    x = torch.tensor(
        sequence,
        dtype=torch.float32,
        device=DEVICE
    )

    x = x.unsqueeze(0)

    return float(
        model(x).item()
    )


# ==========================================================
# METRICS
# ==========================================================

def metrics(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {

        "mean":
            np.mean(values),

        "std":
            np.std(values),

        "rms":
            np.sqrt(
                np.mean(values ** 2)
            ),

        "max_abs":
            np.max(
                np.abs(values)
            ),

    }


# ==========================================================
# RUN ONE SEED
# ==========================================================

def run_seed(
    seed
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"RUNNING SEED {seed}"
    )

    print(
        "=" * 70
    )


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
    # Independent controllers
    # ------------------------------------------------------

    kalman_branch = create_branch()

    ai_branch = create_branch()


    ai_window = []


    kalman_offsets = []
    ai_offsets = []


    for step in range(
        STEPS
    ):

        # --------------------------------------------------
        # ONE environment update
        # --------------------------------------------------

        env = environment.update()


        # --------------------------------------------------
        # ONE noise realization
        # --------------------------------------------------

        noise = noise_model.total_noise(
            env
        )


        # --------------------------------------------------
        # Kalman branch
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
        # AI features
        # --------------------------------------------------

        features = build_features(
            ai_result,
            env,
            noise
        )


        scaled = scaler.transform(
            pd.DataFrame(
                [features],
                columns=FEATURE_COLUMNS
            )
        )[0]


        ai_window.append(
            scaled
        )


        if len(ai_window) > SEQUENCE_LENGTH:

            ai_window.pop(0)


        # --------------------------------------------------
        # AI prediction
        # --------------------------------------------------

        ai_extra_correction = 0.0


        if len(ai_window) == SEQUENCE_LENGTH:

            sequence = np.asarray(
                ai_window,
                dtype=np.float32
            )


            prediction = (
                transformer_predict(
                    sequence
                )
            )


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
        # Record
        # --------------------------------------------------

        kalman_offsets.append(
            kalman_result[
                "corrected_offset"
            ]
        )

        ai_offsets.append(
            ai_branch[
                "oscillator"
            ].measure_error()
        )


        if (
            step + 1
        ) % 1000 == 0:

            print(
                f"{step + 1:5d}/{STEPS} | "
                f"Kalman: "
                f"{kalman_offsets[-1]:+.9e} | "
                f"AI: "
                f"{ai_offsets[-1]:+.9e}"
            )


    # ------------------------------------------------------
    # Remove AI warm-up
    # ------------------------------------------------------

    start = (
        SEQUENCE_LENGTH - 1
    )


    kalman_eval = np.asarray(
        kalman_offsets[
            start:
        ]
    )

    ai_eval = np.asarray(
        ai_offsets[
            start:
        ]
    )


    # ------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------

    kalman_metrics = metrics(
        kalman_eval
    )

    ai_metrics = metrics(
        ai_eval
    )


    rms_improvement = (
        (
            kalman_metrics["rms"]
            -
            ai_metrics["rms"]
        )
        /
        kalman_metrics["rms"]
    ) * 100.0


    std_improvement = (
        (
            kalman_metrics["std"]
            -
            ai_metrics["std"]
        )
        /
        kalman_metrics["std"]
    ) * 100.0


    max_improvement = (
        (
            kalman_metrics["max_abs"]
            -
            ai_metrics["max_abs"]
        )
        /
        kalman_metrics["max_abs"]
    ) * 100.0


    print(
        "\nSeed results:"
    )

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


    return {

        "seed":
            seed,

        "kalman_mean":
            kalman_metrics["mean"],

        "ai_mean":
            ai_metrics["mean"],

        "kalman_std":
            kalman_metrics["std"],

        "ai_std":
            ai_metrics["std"],

        "kalman_rms":
            kalman_metrics["rms"],

        "ai_rms":
            ai_metrics["rms"],

        "kalman_max":
            kalman_metrics["max_abs"],

        "ai_max":
            ai_metrics["max_abs"],

        "rms_improvement_percent":
            rms_improvement,

        "std_improvement_percent":
            std_improvement,

        "max_improvement_percent":
            max_improvement,

    }


# ==========================================================
# RUN ALL SEEDS
# ==========================================================

all_results = []


for seed in SEEDS:

    result = run_seed(
        seed
    )

    all_results.append(
        result
    )


# ==========================================================
# DATAFRAME
# ==========================================================

results_df = pd.DataFrame(
    all_results
)


# ==========================================================
# AGGREGATE STATISTICS
# ==========================================================

mean_rms = (
    results_df[
        "rms_improvement_percent"
    ].mean()
)

std_rms = (
    results_df[
        "rms_improvement_percent"
    ].std(
        ddof=1
    )
)

mean_std = (
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

mean_max = (
    results_df[
        "max_improvement_percent"
    ].mean()
)

std_max = (
    results_df[
        "max_improvement_percent"
    ].std(
        ddof=1
    )
)


# ==========================================================
# SAVE
# ==========================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "multi_seed_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ==========================================================
# PLOT
# ==========================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    results_df["seed"].astype(str),
    results_df[
        "rms_improvement_percent"
    ]
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Random Seed"
)

plt.ylabel(
    "RMS Improvement (%)"
)

plt.title(
    "AI vs Kalman-only RMS Improvement "
    "Across Random Seeds"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "multi_seed_rms_improvement.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


# ==========================================================
# FINAL REPORT
# ==========================================================

print(
    "\n" + "=" * 70
)

print(
    "MULTI-SEED VALIDATION RESULTS"
)

print(
    "=" * 70
)

print(
    "\nPer-seed RMS improvement:"
)

for _, row in results_df.iterrows():

    print(
        f"Seed {int(row['seed']):4d} : "
        f"{row['rms_improvement_percent']:+.2f}%"
    )


print(
    "\nPer-seed STD improvement:"
)

for _, row in results_df.iterrows():

    print(
        f"Seed {int(row['seed']):4d} : "
        f"{row['std_improvement_percent']:+.2f}%"
    )


print(
    "\nPer-seed maximum-offset improvement:"
)

for _, row in results_df.iterrows():

    print(
        f"Seed {int(row['seed']):4d} : "
        f"{row['max_improvement_percent']:+.2f}%"
    )


print(
    "\n" + "-" * 70
)

print(
    "AGGREGATE"
)

print(
    "-" * 70
)

print(
    f"RMS improvement : "
    f"{mean_rms:+.2f}% ± {std_rms:.2f}%"
)

print(
    f"STD improvement : "
    f"{mean_std:+.2f}% ± {std_std:.2f}%"
)

print(
    f"MAX improvement : "
    f"{mean_max:+.2f}% ± {std_max:.2f}%"
)


print(
    "\nSaved:"
)

print(
    csv_path
)

print(
    plot_path
)

print(
    "\n[PASS] Multi-seed AI vs "
    "Kalman validation completed."
)