"""
tests/test_noise.py
====================
Manual validation of NoiseModel — verifies each component behaves
as expected over 20 iterations.

Run from the project root:
    python tests/test_noise.py
    python3 tests/test_noise.py
    py tests/test_noise.py
"""

import sys
import os

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.noise import NoiseModel

noise = NoiseModel(seed=42)

environment = {
    "temperature": 300.0,
    "magnetic": 50e-6,
    "time": 0.0,
    "elapsed_time": 0.0
}

print("=" * 90)
print("Noise Model Validation")
print("=" * 90)

for i in range(20):

    environment["time"] = float(i + 1)
    environment["elapsed_time"] = float(i + 1)
    values = noise.total_noise(environment)

    print(f"\nIteration {i+1}")

    if isinstance(values, dict):
        for k, v in values.items():
            print(f"{k:20s}: {v:.10e}")
    else:
        print(values)

# ------------------------------------------------------------------
# Sanity check: systematic sources must be non-zero off-nominal
# ------------------------------------------------------------------
print()
print("=" * 90)
print("Sanity Check -- Systematic sources at OFF-NOMINAL conditions")
print("=" * 90)

off_nominal = {
    "temperature": 305.0,   # +5 K above reference
    "magnetic":    55e-6,   # +5 uT above reference
    "time": 0.0,
    "elapsed_time": 0.0,
}

noise_check = NoiseModel(seed=0)
r = noise_check.total_noise(off_nominal)

print(f"\n  temperature_shift  : {r['temperature']:+.6e}  (expect > 0, T > T_ref)")
print(f"  zeeman_shift       : {r['zeeman']:+.6e}  (expect > 0, B > B_ref)")
print(f"  blackbody_shift    : {r['blackbody']:+.6e}  (expect > 0, T > T_ref)")

assert r["temperature"] > 0, "temperature_shift should be positive when T > T_ref"
assert r["zeeman"]      > 0, "zeeman_shift should be positive when B > B_ref"
assert r["blackbody"]   > 0, "blackbody_shift should be positive when T > T_ref"

print()
print("  [PASS] All systematic sources respond correctly to off-nominal inputs.")
print()
print("=" * 90)
print("  NOTE: temperature / zeeman / blackbody = 0 in the 20-step loop above")
print("  because the environment was AT the reference point (T=300 K, B=50 uT).")
print("  Zero deviation => zero systematic shift. This is CORRECT behaviour.")
print("=" * 90)
