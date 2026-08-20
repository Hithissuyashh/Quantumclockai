"""
validate_environment.py
========================
Phase 1.1 - Validation script for LaboratoryEnvironment.

Run from the QuantumClockAI project root:
    python validate_environment.py

Expected: small, smooth changes between timesteps - no abrupt jumps.
"""

import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from simulator.environment import LaboratoryEnvironment


def _fmt_row(state: dict) -> str:
    """Format one state dict as a readable table row."""
    return (
        f"{int(state['time']):>4d}"
        f"{state['temperature']:>9.4f} K  |  "
        f"{state['magnetic']*1e6:>9.4f} uT  |  "
        f"{state['laser_power']:>10.6f}  |  "
        f"{state['pressure']:>12.3f} Pa  |  "
        f"{state['humidity']:>8.4f} %"
    )


def main() -> None:
    SEP = "=" * 88
    DASH = "-" * 88

    print()
    print(SEP)
    print("  Phase 1.1 -- LaboratoryEnvironment Validation")
    print(SEP)

    header = (
        f"  {'Step':>4}  |  "
        f"{'Temperature':>11}  |  "
        f"{'Magnetic':>11}  |  "
        f"{'Laser Power':>12}  |  "
        f"{'Pressure':>14}  |  "
        f"{'Humidity':>10}"
    )

    print(header)
    print(DASH)



    env = LaboratoryEnvironment(seed=42)

    prev = None
    for _ in range(10):
        env.update()
        state = env.get_state()
        print(_fmt_row(state))

        # Smoothness assertions
        if prev is not None:
            assert abs(state["temperature"] - prev["temperature"]) < 1.0,   \
                "Temperature jump too large!"
            assert abs(state["magnetic"]    - prev["magnetic"])    < 5e-6,  \
                "Magnetic field jump too large!"
            assert abs(state["laser_power"] - prev["laser_power"]) < 0.05, \
                "Laser power jump too large!"
            assert abs(state["pressure"]    - prev["pressure"])    < 50.0,  \
                "Pressure jump too large!"
            assert abs(state["humidity"]    - prev["humidity"])    < 2.0,   \
                "Humidity jump too large!"
        prev = state

    print(DASH)
    print()
    print("  [PASS] All smoothness assertions passed - no abrupt jumps detected.")

    # Reset test
    print()
    print("  Testing reset()...")
    env.reset()
    s = env.get_state()
    assert s["time"]        == 0,           "time should be 0 after reset"
    assert s["temperature"] == 300.0,       "temperature should reset to 300 K"
    assert s["magnetic"]    == 50e-6,       "magnetic should reset to 50 uT"
    assert s["laser_power"] == 1.0,         "laser_power should reset to 1.0"
    assert s["pressure"]    == 101_325.0,   "pressure should reset to 101325 Pa"
    assert s["humidity"]    == 45.0,        "humidity should reset to 45 %"
    print("  [PASS] reset() restores all parameters to nominal values.")

    # ── Reproducibility test ────────────────────────────────────────
    print()
    print("  Testing seed reproducibility...")
    env_a = LaboratoryEnvironment(seed=42)
    for _ in range(5):
        env_a.update()
    sa = env_a.get_state()

    env_b = LaboratoryEnvironment(seed=42)
    for _ in range(5):
        env_b.update()
    sb = env_b.get_state()
    assert sa["temperature"] == sb["temperature"], "seed=42 should give identical temperature"
    assert sa["magnetic"]    == sb["magnetic"],    "seed=42 should give identical magnetic"
    assert sa["pressure"]    == sb["pressure"],    "seed=42 should give identical pressure"
    print("  [PASS] Two runs with seed=42 produce identical trajectories.")

    # ── dt test ─────────────────────────────────────────────────────
    print()
    print("  Testing dt timestep attribute...")
    env_dt = LaboratoryEnvironment(seed=0, dt=0.1)
    assert env_dt.dt == 0.1, "dt should be stored as 0.1"
    env_dt.update()
    state_dt = env_dt.get_state()
    assert state_dt["dt"] == 0.1,            "get_state() should expose dt"
    assert abs(state_dt["elapsed_time"] - 0.1) < 1e-12, \
        "elapsed_time should equal 1 * dt = 0.1 s"
    env_dt.update()
    assert abs(env_dt.get_state()["elapsed_time"] - 0.2) < 1e-12, \
        "elapsed_time should equal 2 * dt = 0.2 s"
    print(f"  [PASS] dt=0.1 s stored correctly; elapsed_time tracks steps x dt.")
    print()

    # Repr test
    env.update()
    print("  repr:", repr(env))
    print()


if __name__ == "__main__":
    main()
