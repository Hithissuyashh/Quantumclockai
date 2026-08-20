import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulator.clock import QuantumClock


STEPS = 10_000
DT = 1.0


def run_clock(use_servo):

    clock = QuantumClock(seed=42)

    offsets = []

    for _ in range(STEPS):

        result = clock.step()

        if use_servo:
            offsets.append(
                result["corrected_offset"]
            )
        else:
            offsets.append(
                result["true_detuning"]
            )

            # Cancel the correction so the oscillator
            # remains genuinely open-loop.
            clock.oscillator.offset = (
                result["true_detuning"]
            )

    return np.asarray(
        offsets,
        dtype=np.float64
    )


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
    deviations = []

    for m in m_values:

        if 2 * m >= n:
            continue

        usable = (n // m) * m

        blocks = data[:usable].reshape(
            -1,
            m
        )

        averages = blocks.mean(axis=1)

        if len(averages) < 2:
            continue

        diff = np.diff(averages)

        variance = 0.5 * np.mean(
            diff ** 2
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


print("=" * 70)
print("OPEN-LOOP vs CLOSED-LOOP QUANTUM CLOCK")
print("=" * 70)

print("\nRunning OPEN-LOOP experiment...")

open_loop = run_clock(
    use_servo=False
)

print("Running CLOSED-LOOP experiment...")

closed_loop = run_clock(
    use_servo=True
)


# -------------------------------------------------
# RMS comparison
# -------------------------------------------------

rms_open = np.sqrt(
    np.mean(open_loop ** 2)
)

rms_closed = np.sqrt(
    np.mean(closed_loop ** 2)
)

std_open = np.std(open_loop)
std_closed = np.std(closed_loop)

improvement = (
    1.0 -
    rms_closed / rms_open
) * 100


print("\n" + "=" * 70)
print("STABILITY RESULTS")
print("=" * 70)

print(
    f"Open-loop RMS   : "
    f"{rms_open:.12e} Hz"
)

print(
    f"Closed-loop RMS : "
    f"{rms_closed:.12e} Hz"
)

print(
    f"\nOpen-loop STD   : "
    f"{std_open:.12e} Hz"
)

print(
    f"Closed-loop STD : "
    f"{std_closed:.12e} Hz"
)

print(
    f"\nRMS improvement : "
    f"{improvement:+.2f}%"
)


# -------------------------------------------------
# Allan deviation
# -------------------------------------------------

tau_open, adev_open = allan_deviation(
    open_loop,
    DT
)

tau_closed, adev_closed = allan_deviation(
    closed_loop,
    DT
)


comparison = pd.DataFrame({
    "tau_seconds": tau_open,
    "allan_open_loop_hz": adev_open,
    "allan_closed_loop_hz": adev_closed,
})

comparison.to_csv(
    "results/open_vs_closed_allan.csv",
    index=False
)


# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(9, 6))

plt.loglog(
    tau_open,
    adev_open,
    marker="o",
    markersize=4,
    label="Open Loop"
)

plt.loglog(
    tau_closed,
    adev_closed,
    marker="s",
    markersize=4,
    label="Closed Loop"
)

plt.xlabel(
    "Averaging Time τ (s)"
)

plt.ylabel(
    "Allan Deviation (Hz)"
)

plt.title(
    "Quantum Clock Stability: Open Loop vs Closed Loop"
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/open_vs_closed_allan.png",
    dpi=300
)

plt.close()


print("\nSaved:")
print("results/open_vs_closed_allan.csv")
print("results/open_vs_closed_allan.png")

print(
    "\n[PASS] Open-loop vs closed-loop "
    "experiment completed."
)