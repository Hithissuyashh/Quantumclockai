import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "results/allan_deviation.csv"
OUTPUT_FILE = "results/fractional_allan_deviation.csv"
PLOT_FILE = "results/fractional_allan_deviation.png"

# Sr-87 atomic reference frequency
F0 = 429228004229873.0  # Hz

df = pd.read_csv(INPUT_FILE)

tau = df["tau_seconds"].to_numpy()
allan_hz = df["allan_deviation_hz"].to_numpy()

# Convert absolute frequency stability to fractional stability
fractional_allan = allan_hz / F0

result = pd.DataFrame({
    "tau_seconds": tau,
    "allan_deviation_hz": allan_hz,
    "fractional_allan_deviation": fractional_allan,
})

result.to_csv(
    OUTPUT_FILE,
    index=False
)

# Plot
plt.figure(figsize=(8, 5))

plt.loglog(
    tau,
    fractional_allan,
    marker="o",
    markersize=4
)

plt.xlabel("Averaging Time τ (s)")
plt.ylabel("Fractional Allan Deviation σᵧ(τ)")
plt.title("Quantum Clock Fractional Frequency Stability")

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=300
)

plt.close()

# Report minimum
i = np.argmin(fractional_allan)

print("=" * 70)
print("FRACTIONAL ALLAN DEVIATION")
print("=" * 70)

print(f"Reference frequency : {F0:.6e} Hz")
print(f"Minimum τ           : {tau[i]:.1f} s")
print(
    f"Minimum σy(τ)       : "
    f"{fractional_allan[i]:.12e}"
)

print("\nSaved:")
print(OUTPUT_FILE)
print(PLOT_FILE)