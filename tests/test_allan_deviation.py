import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "results/quantum_clock_stability.csv"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------
# Load stability data
# -------------------------------------------------

df = pd.read_csv(INPUT_FILE)

frequency_error = df["true_detuning"].to_numpy(
    dtype=np.float64
)

dt = 1.0  # seconds


# -------------------------------------------------
# Allan deviation
# -------------------------------------------------

def allan_deviation(data, dt):

    n = len(data)

    max_m = n // 4

    # Log-spaced averaging factors
    m_values = np.unique(
        np.logspace(
            0,
            np.log10(max_m),
            30
        ).astype(int)
    )

    taus = []
    deviations = []

    for m in m_values:

        if 2 * m >= n:
            continue

        # Average consecutive blocks
        usable = (n // m) * m

        blocks = data[:usable].reshape(
            -1,
            m
        )

        averages = blocks.mean(axis=1)

        if len(averages) < 2:
            continue

        # Allan variance
        diff = np.diff(averages)

        variance = (
            0.5 *
            np.mean(diff ** 2)
        )

        if variance <= 0:
            continue

        taus.append(m * dt)
        deviations.append(
            np.sqrt(variance)
        )

    return (
        np.asarray(taus),
        np.asarray(deviations)
    )


taus, adev = allan_deviation(
    frequency_error,
    dt
)


# -------------------------------------------------
# Save numerical results
# -------------------------------------------------

allan_df = pd.DataFrame({
    "tau_seconds": taus,
    "allan_deviation_hz": adev,
})

output_csv = os.path.join(
    OUTPUT_DIR,
    "allan_deviation.csv"
)

allan_df.to_csv(
    output_csv,
    index=False
)


# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.loglog(
    taus,
    adev,
    marker="o",
    markersize=4
)

plt.xlabel(
    "Averaging Time τ (s)"
)

plt.ylabel(
    "Allan Deviation (Hz)"
)

plt.title(
    "Quantum Clock Frequency Stability"
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.tight_layout()

output_plot = os.path.join(
    OUTPUT_DIR,
    "allan_deviation.png"
)

plt.savefig(
    output_plot,
    dpi=300
)

plt.close()


# -------------------------------------------------
# Report
# -------------------------------------------------

print("=" * 70)
print("ALLAN DEVIATION ANALYSIS")
print("=" * 70)

print(
    f"Samples              : {len(frequency_error)}"
)

print(
    f"Sampling interval    : {dt:.1f} s"
)

print(
    f"Maximum averaging τ  : "
    f"{taus[-1]:.1f} s"
)

print(
    f"\nMinimum Allan deviation:"
)

min_index = np.argmin(adev)

print(
    f"τ    : {taus[min_index]:.1f} s"
)

print(
    f"σᵧ   : {adev[min_index]:.12e} Hz"
)

print(
    f"\nSaved:"
)

print(output_csv)
print(output_plot)