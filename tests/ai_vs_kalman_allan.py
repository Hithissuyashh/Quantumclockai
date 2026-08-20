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
DT = 1.0

AI_GAIN = 0.05
AI_CORRECTION_LIMIT = 0.0002

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

SCALER_PATH = (
    "data/processed/feature_scaler_v3.pkl"
)

OUTPUT_DIR = (
    "results/ai_vs_kalman_allan"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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
# LOAD SCALER
# ==========================================================

print("=" * 70)
print("AI vs KALMAN-ONLY ALLAN DEVIATION")
print("=" * 70)

print(f"\nDevice : {DEVICE}")
print(f"Seeds  : {SEEDS}")
print(f"Steps  : {STEPS}")
print(f"AI gain: {AI_GAIN}")

scaler = joblib.load(
    SCALER_PATH
)

print("✓ v3 scaler loaded.")


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

print("✓ Transformer loaded.")


# ==========================================================
# CONTROLLER BRANCH
# ==========================================================

def create_branch():

    return {
        "oscillator": OpticalOscillator(),

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


    return {
        "true_detuning":
            true_detuning,

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
            oscillator.measure_error(),
    }


# ==========================================================
# FEATURE CONSTRUCTION
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
# ALLAN DEVIATION
# ==========================================================

def allan_deviation(
    frequency_data,
    dt=1.0
):

    y = np.asarray(
        frequency_data,
        dtype=np.float64
    )

    n = len(y)

    taus = []

    deviations = []

    # Logarithmically spaced averaging factors
    max_m = n // 4

    m_values = np.unique(
        np.logspace(
            0,
            np.log10(max_m),
            30
        ).astype(int)
    )

    m_values = m_values[
        m_values >= 1
    ]


    for m in m_values:

        if 2 * m >= n:
            continue


        # Number of complete blocks
        k = n // m

        truncated = y[
            :k * m
        ]


        # Average each block
        block_means = (
            truncated
            .reshape(k, m)
            .mean(axis=1)
        )


        if len(block_means) < 2:
            continue


        # Allan variance
        variance = (
            0.5 *
            np.mean(
                np.diff(
                    block_means
                ) ** 2
            )
        )


        if variance < 0:
            continue


        taus.append(
            m * dt
        )

        deviations.append(
            np.sqrt(variance)
        )


    return (
        np.asarray(taus),
        np.asarray(deviations)
    )


# ==========================================================
# RUN ONE SEED
# ==========================================================

def run_seed(seed):

    print(
        "\n" + "=" * 70
    )

    print(
        f"RUNNING SEED {seed}"
    )

    print(
        "=" * 70
    )


    environment = LaboratoryEnvironment(
        seed=seed
    )

    noise_model = NoiseModel(
        seed=seed
    )


    kalman_branch = create_branch()

    ai_branch = create_branch()


    ai_window = []

    kalman_offsets = []

    ai_offsets = []


    for step in range(
        STEPS
    ):

        # --------------------------------------------------
        # ONE shared physical realization
        # --------------------------------------------------

        env = environment.update()

        noise = noise_model.total_noise(
            env
        )


        # --------------------------------------------------
        # Kalman-only
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
        # AI feature vector
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
        # AI correction
        # --------------------------------------------------

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


            correction = (
                AI_GAIN *
                prediction
            )


            correction = float(
                np.clip(
                    correction,
                    -AI_CORRECTION_LIMIT,
                    AI_CORRECTION_LIMIT
                )
            )


            ai_branch[
                "oscillator"
            ].apply_correction(
                correction
            )


        # --------------------------------------------------
        # Record final offsets
        # --------------------------------------------------

        kalman_offsets.append(
            kalman_branch[
                "oscillator"
            ].measure_error()
        )

        ai_offsets.append(
            ai_branch[
                "oscillator"
            ].measure_error()
        )


        if (
            step + 1
        ) % 2000 == 0:

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


    kalman_offsets = np.asarray(
        kalman_offsets[start:]
    )

    ai_offsets = np.asarray(
        ai_offsets[start:]
    )


    # ------------------------------------------------------
    # Allan deviation
    # ------------------------------------------------------

    tau_k, allan_k = allan_deviation(
        kalman_offsets,
        DT
    )

    tau_ai, allan_ai = allan_deviation(
        ai_offsets,
        DT
    )


    # Use common τ values
    common_tau = np.intersect1d(
        tau_k,
        tau_ai
    )


    kalman_map = dict(
        zip(
            tau_k,
            allan_k
        )
    )

    ai_map = dict(
        zip(
            tau_ai,
            allan_ai
        )
    )


    rows = []

    for tau in common_tau:

        k_value = kalman_map[tau]

        ai_value = ai_map[tau]

        improvement = (
            (
                k_value -
                ai_value
            )
            /
            k_value
        ) * 100.0


        rows.append({

            "seed":
                seed,

            "tau_seconds":
                tau,

            "allan_kalman_hz":
                k_value,

            "allan_ai_hz":
                ai_value,

            "improvement_percent":
                improvement,

        })


    return pd.DataFrame(rows)


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


allan_df = pd.concat(
    all_results,
    ignore_index=True
)


# ==========================================================
# AGGREGATE ACROSS SEEDS
# ==========================================================

summary = (
    allan_df
    .groupby(
        "tau_seconds"
    )
    .agg(

        allan_kalman_hz_mean=(
            "allan_kalman_hz",
            "mean"
        ),

        allan_kalman_hz_std=(
            "allan_kalman_hz",
            "std"
        ),

        allan_ai_hz_mean=(
            "allan_ai_hz",
            "mean"
        ),

        allan_ai_hz_std=(
            "allan_ai_hz",
            "std"
        ),

        improvement_percent_mean=(
            "improvement_percent",
            "mean"
        ),

        improvement_percent_std=(
            "improvement_percent",
            "std"
        ),

    )
    .reset_index()
)


# ==========================================================
# SAVE DATA
# ==========================================================

raw_path = os.path.join(
    OUTPUT_DIR,
    "allan_multiseed_raw.csv"
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "allan_multiseed_summary.csv"
)

allan_df.to_csv(
    raw_path,
    index=False
)

summary.to_csv(
    summary_path,
    index=False
)


# ==========================================================
# PLOT — ALLAN DEVIATION
# ==========================================================

plt.figure(
    figsize=(10, 7)
)

plt.errorbar(
    summary["tau_seconds"],
    summary[
        "allan_kalman_hz_mean"
    ],
    yerr=summary[
        "allan_kalman_hz_std"
    ],
    marker="o",
    capsize=3,
    label="Kalman-only"
)

plt.errorbar(
    summary["tau_seconds"],
    summary[
        "allan_ai_hz_mean"
    ],
    yerr=summary[
        "allan_ai_hz_std"
    ],
    marker="o",
    capsize=3,
    label="AI-assisted"
)

plt.xscale(
    "log"
)

plt.yscale(
    "log"
)

plt.xlabel(
    "Averaging Time τ (s)"
)

plt.ylabel(
    "Allan Deviation (Hz)"
)

plt.title(
    "AI-assisted vs Kalman-only "
    "Quantum Clock Stability"
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "ai_vs_kalman_allan_deviation.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


# ==========================================================
# PLOT — IMPROVEMENT
# ==========================================================

plt.figure(
    figsize=(10, 6)
)

plt.errorbar(
    summary["tau_seconds"],
    summary[
        "improvement_percent_mean"
    ],
    yerr=summary[
        "improvement_percent_std"
    ],
    marker="o",
    capsize=3
)

plt.axhline(
    0,
    linewidth=1
)

plt.xscale(
    "log"
)

plt.xlabel(
    "Averaging Time τ (s)"
)

plt.ylabel(
    "Allan Deviation Improvement (%)"
)

plt.title(
    "AI Stability Improvement "
    "Across Averaging Times"
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.tight_layout()

improvement_path = os.path.join(
    OUTPUT_DIR,
    "allan_improvement_vs_tau.png"
)

plt.savefig(
    improvement_path,
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
    "AI vs KALMAN ALLAN DEVIATION RESULTS"
)

print(
    "=" * 70
)


print(
    "\nAveraging-time comparison:"
)

for _, row in summary.iterrows():

    print(
        f"τ = "
        f"{row['tau_seconds']:8.1f} s | "
        f"Kalman = "
        f"{row['allan_kalman_hz_mean']:.6e} Hz | "
        f"AI = "
        f"{row['allan_ai_hz_mean']:.6e} Hz | "
        f"Improvement = "
        f"{row['improvement_percent_mean']:+.2f}%"
    )


print(
    "\nSaved:"
)

print(
    raw_path
)

print(
    summary_path
)

print(
    plot_path
)

print(
    improvement_path
)


print(
    "\n[PASS] AI vs Kalman Allan deviation "
    "experiment completed."
)