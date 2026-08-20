import csv
import os
import numpy as np

from simulator.clock import QuantumClock


clock = QuantumClock(seed=42)

steps = 10_000

results = []

print("=" * 70)
print("QUANTUM CLOCK LONG-RUN STABILITY TEST")
print("=" * 70)

for i in range(steps):

    result = clock.step()

    # Store every value returned by the clock
    row = {
        "step": i + 1,
        **result,
    }

    results.append(row)

    if (i + 1) % 1000 == 0:
        print(
            f"{i + 1:5d} / {steps} | "
            f"Offset: {result['true_detuning']:+.9f} Hz"
        )


# -------------------------------------------------
# Extract true frequency offset
# -------------------------------------------------

offsets_before = np.asarray(
    [r["true_detuning"] for r in results],
    dtype=np.float64,
)

offsets_after = np.asarray(
    [r["corrected_offset"] for r in results],
    dtype=np.float64,
)

offsets = offsets_after

mean_offset = np.mean(offsets)
std_offset = np.std(offsets)
rms_offset = np.sqrt(np.mean(offsets ** 2))
max_offset = np.max(np.abs(offsets))


# -------------------------------------------------
# Save complete stability history
# -------------------------------------------------

os.makedirs("results", exist_ok=True)

output_file = "results/quantum_clock_stability.csv"

# Collect all keys returned by clock.step()
fieldnames = ["step"]

for row in results:
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)


with open(
    output_file,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results)


# -------------------------------------------------
# Results
# -------------------------------------------------

print("\n" + "=" * 70)
print("STABILITY RESULTS")
print("=" * 70)

print(
    f"Mean offset       : "
    f"{mean_offset:+.12e} Hz"
)

print(
    f"Std deviation     : "
    f"{std_offset:.12e} Hz"
)

print(
    f"RMS offset        : "
    f"{rms_offset:.12e} Hz"
)

print(
    f"Maximum offset    : "
    f"{max_offset:.12e} Hz"
)

print("\nSaved stability data:")
print(output_file)


# -------------------------------------------------
# Basic stability checks
# -------------------------------------------------

assert np.isfinite(mean_offset)
assert np.isfinite(std_offset)
assert np.isfinite(rms_offset)
assert np.isfinite(max_offset)

assert max_offset < 10.0

print(
    "\n[PASS] "
    "10,000-step quantum clock stability test passed."
)