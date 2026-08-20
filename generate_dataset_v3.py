import pandas as pd

from simulator.clock import QuantumClock


N = 100_000

clock = QuantumClock(seed=42)

rows = []

print("=" * 70)
print("GENERATING QUANTUM CLOCK DATASET V3")
print("=" * 70)

for i in range(N):

    result = clock.step()

    env = result["environment"]
    noise = result["noise"]

    row = {
        # Time
        "time": result["time"],

        # Environment
        "temperature": env["temperature"],
        "magnetic": env["magnetic"],
        "laser_power": env["laser_power"],
        "pressure": env["pressure"],
        "humidity": env["humidity"],

        # Noise
        "white": noise["white"],
        "random_walk": noise["random_walk"],
        "flicker": noise["flicker"],
        "laser_phase": noise["laser_phase"],
        "qpn": noise["qpn"],
        "temperature_shift": noise["temperature"],
        "zeeman": noise["zeeman"],
        "blackbody": noise["blackbody"],
        "aging": noise["aging"],
        "total_noise": noise["total"],

        # Quantum reference
        "atomic_frequency":
            result["atomic_frequency"],

        "true_detuning":
            result["true_detuning"],

        "excitation_probability_plus":
            result["excitation_probability_plus"],

        "excitation_probability_minus":
            result["excitation_probability_minus"],

        # Clock measurement/control
        "measured_frequency":
            result["measured_frequency"],

        "measured_offset":
            result["measured_offset"],

        "estimated_offset":
            result["estimated_offset"],

        "servo_correction":
            result["servo_correction"],

        "fractional_frequency":
            result["fractional_frequency"],
    }

    rows.append(row)

    if i % 10_000 == 0:
        print(f"{i:,} / {N:,} steps completed")


df = pd.DataFrame(rows)

output = "data/raw/clock_dataset_v3.csv"

df.to_csv(output, index=False)

print("\n" + "=" * 70)
print("DATASET GENERATION COMPLETE")
print("=" * 70)

print(df.head())

print("\nShape :", df.shape)

print("\nSaved :", output)
