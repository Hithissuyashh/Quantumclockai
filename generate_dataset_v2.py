import os
import pandas as pd

from simulator.clock import QuantumClock

# ----------------------------------
# Settings
# ----------------------------------

NUM_STEPS = 100000
OUTPUT_DIR = "data"
OUTPUT_FILE = "clock_dataset_v2.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

clock = QuantumClock(seed=42)

rows = []

print("Generating Dataset V2...\n")

for i in range(NUM_STEPS):

    result = clock.step()

    env = result["environment"]
    noise = result["noise"]

    rows.append({

        # Time
        "time": result["time"],

        # Environment
        "temperature": env["temperature"],
        "magnetic": env["magnetic"],
        "laser_power": env["laser_power"],
        "pressure": env["pressure"],
        "humidity": env["humidity"],

        # Individual Noise Sources
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

        # Clock State
        "true_offset": result["true_offset"],
        "measured_offset": result["measured_offset"],
        "estimated_offset": result["estimated_offset"],
        "servo_correction": result["servo_correction"],
        "fractional_frequency": result["fractional_frequency"]

    })

    if (i + 1) % 10000 == 0:
        print(f"{i+1:,} / {NUM_STEPS:,} steps completed")

df = pd.DataFrame(rows)

path = os.path.join(
    OUTPUT_DIR,
    OUTPUT_FILE
)

df.to_csv(path, index=False)

print("\n===================================")
print("Dataset generation complete!")
print("===================================")
print(df.head())
print("\nShape :", df.shape)
print("Saved :", path)
