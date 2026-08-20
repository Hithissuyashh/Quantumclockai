import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FILE = "results/quantum_clock_stability.csv"

df = pd.read_csv(FILE)

before = df["true_detuning"].to_numpy(dtype=np.float64)
after = df["corrected_offset"].to_numpy(dtype=np.float64)

dt = 1.0


def allan_deviation(data, dt):

    n = len(data)

    m_values = np.unique(
        np.logspace(
            0,
            np.log10(n // 4),
            30
        ).astype(int)
    )

    taus = []
    adev = []

    for m in m_values:

        if 2 * m >= n:
            continue

        usable = (n // m) * m

        blocks = data[:usable].reshape(-1, m)

        averages = blocks.mean(axis=1)

        if len(averages) < 2:
            continue

        diff = np.diff(averages)

        variance = 0.5 * np.mean(diff ** 2)

        if variance <= 0:
            continue

        taus.append(m * dt)
        adev.append(np.sqrt(variance))

    return np.array(taus), np.array(adev)


tau_before, adev_before = allan_deviation(
    before,
    dt
)

tau_after, adev_after = allan_deviation(
    after,
    dt
)


# -------------------------------------------------
# Metrics
# -------------------------------------------------

rms_before = np.sqrt(np.mean(before ** 2))
rms_after = np.sqrt(np.mean(after ** 2))

std_before = np.std(before)
std_after = np.std(after)

print("=" * 70)
print("BEFORE vs AFTER SERVO STABILITY")
print("=" * 70)

print(f"Before RMS : {rms_before:.12e} Hz")
print(f"After RMS  : {rms_after:.12e} Hz")

print(f"\nBefore STD : {std_before:.12e} Hz")
print(f"After STD  : {std_after:.12e} Hz")

improvement = (
    1.0 -
    rms_after / rms_before
) * 100

print(
    f"\nRMS change : {improvement:+.2f}%"
)


# -------------------------------------------------
# Save comparison
# -------------------------------------------------

comparison = pd.DataFrame({
    "tau_seconds": tau_before,
    "allan_before_hz": adev_before,
    "allan_after_hz": adev_after,
})

comparison.to_csv(
    "results/stability_comparison.csv",
    index=False
)


# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(8, 5))

plt.loglog(
    tau_before,
    adev_before,
    marker="o",
    markersize=4,
    label="Before Servo"
)

plt.loglog(
    tau_after,
    adev_after,
    marker="s",
    markersize=4,
    label="After Servo"
)

plt.xlabel("Averaging Time τ (s)")
plt.ylabel("Allan Deviation (Hz)")
plt.title("Quantum Clock Stability: Before vs After Servo")

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/stability_before_vs_after.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("results/stability_comparison.csv")
print("results/stability_before_vs_after.png")