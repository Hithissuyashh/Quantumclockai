"""
validate_noise.py
=================
Phase 1.2 - Validation script for NoiseModel.

Run from the QuantumClockAI project root:
    python validate_noise.py

Tests:
  1. Basic total_noise() output with all sources active
  2. Dictionary keys and total = sum of components
  3. Accumulation: random_walk and laser_phase grow over time
  4. Ablation: disabling sources zeroes their contribution
  5. enable() / disable() chaining and active_sources()
  6. reset_state() clears accumulators
  7. Reproducibility via seed
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from simulator.noise import NoiseModel, ALL_SOURCES
from simulator.environment import LaboratoryEnvironment


SEP  = "=" * 70
DASH = "-" * 70
EXPECTED_KEYS = {
    "white", "random_walk", "flicker", "laser_phase",
    "qpn", "temperature", "zeeman", "blackbody", "aging", "total",
}


def make_env(t=300.0, b=50e-6):
    return {
        "temperature": t,
        "magnetic": b,
        "time": 0.0,
        "elapsed_time": 0.0
    }


def main():
    print()
    print(SEP)
    print("  Phase 1.2 -- NoiseModel Validation")
    print(SEP)

    # ------------------------------------------------------------------
    # 1. Basic output: 10 steps, all sources active
    # ------------------------------------------------------------------
    print("\n  [1] 10-step output (all sources active)")
    print(DASH)
    header = (
        f"  {'Step':>4}  {'white':>10}  {'rw':>12}  "
        f"{'flicker':>10}  {'laser':>12}  {'total':>14}"
    )
    print(header)
    print(DASH)

    noise = NoiseModel(seed=42)
    env_state = make_env()

    for i in range(1, 11):
        env_state["time"] = float(i)
        env_state["elapsed_time"] = float(i)
        r = noise.total_noise(env_state)
        print(
            f"  {i:>4}  {r['white']:>10.6f}  {r['random_walk']:>12.8f}  "
            f"{r['flicker']:>10.6f}  {r['laser_phase']:>12.8f}  {r['total']:>14.8f}"
        )

    # ------------------------------------------------------------------
    # 2. Dictionary completeness and total = sum of components
    # ------------------------------------------------------------------
    print(f"\n  [2] Dict keys and total integrity")
    noise2 = NoiseModel(seed=0)
    r = noise2.total_noise(make_env())
    assert set(r.keys()) == EXPECTED_KEYS, f"Missing keys: {EXPECTED_KEYS - set(r.keys())}"

    component_sum = sum(v for k, v in r.items() if k != "total")
    assert abs(r["total"] - component_sum) < 1e-15, \
        f"total {r['total']} != sum of components {component_sum}"
    print("  [PASS] All 9 keys present; total == sum of components.")

    # ------------------------------------------------------------------
    # 3. Accumulation check
    # ------------------------------------------------------------------
    print(f"\n  [3] Accumulation: random_walk and laser_phase drift over time")
    noise3 = NoiseModel(seed=7)
    rw_vals, lp_vals = [], []
    for _ in range(50):
        r = noise3.total_noise(make_env())
        rw_vals.append(r["random_walk"])
        lp_vals.append(r["laser_phase"])

    rw_range = max(rw_vals) - min(rw_vals)
    lp_range = max(lp_vals) - min(lp_vals)
    assert rw_range > 0, "random_walk should accumulate (non-zero range)"
    assert lp_range > 0, "laser_phase should accumulate (non-zero range)"
    print(f"  [PASS] random_walk range over 50 steps: {rw_range:.6e}")
    print(f"  [PASS] laser_phase  range over 50 steps: {lp_range:.6e}")

    # ------------------------------------------------------------------
    # 4. Ablation: disable a source -> its contribution is 0
    # ------------------------------------------------------------------
    print(f"\n  [4] Ablation study: disable individual sources")
    for src in sorted(ALL_SOURCES):
        n = NoiseModel(seed=1)
        n.disable(src)
        r = n.total_noise(make_env())
        assert r[src] == 0.0, f"Disabled source '{src}' should be 0.0, got {r[src]}"
    print("  [PASS] Every source contributes exactly 0.0 when disabled.")

    # ------------------------------------------------------------------
    # 5. enable/disable chaining and active_sources()
    # ------------------------------------------------------------------
    print(f"\n  [5] enable() / disable() chaining and active_sources()")
    n = NoiseModel(seed=0, enabled={"white", "random_walk"})
    assert n.active_sources() == {"white", "random_walk"}, "Wrong initial enabled set"
    n.enable("flicker").disable("white")
    assert n.active_sources() == {"random_walk", "flicker"}, \
        f"Wrong enabled set after chaining: {n.active_sources()}"
    print("  [PASS] Chaining enable().disable() works correctly.")
    print(f"         Active sources: {sorted(n.active_sources())}")

    # ------------------------------------------------------------------
    # 6. reset_state() clears accumulators
    # ------------------------------------------------------------------
    print(f"\n  [6] reset_state() clears accumulated processes")
    n = NoiseModel(seed=5)
    for _ in range(20):
        n.total_noise(make_env())
    snap_before = n.state_snapshot()
    n.reset_state()
    snap_after = n.state_snapshot()
    assert snap_after["random_walk_state"] == 0.0
    assert snap_after["flicker_state"]     == 0.0
    assert snap_after["laser_phase"]       == 0.0
    print(f"  [PASS] Before reset: rw={snap_before['random_walk_state']:.4e}, "
          f"lp={snap_before['laser_phase']:.4e}")
    print(f"         After  reset: rw=0.0, lp=0.0")

    # ------------------------------------------------------------------
    # 7. Reproducibility
    # ------------------------------------------------------------------
    print(f"\n  [7] Reproducibility via seed")
    def run_n(seed, steps=10):
        n = NoiseModel(seed=seed)
        totals = []
        for _ in range(steps):
            totals.append(n.total_noise(make_env())["total"])
        return totals

    run_a = run_n(99)
    run_b = run_n(99)
    assert run_a == run_b, "Same seed should give identical total sequence"
    print("  [PASS] Two runs with seed=99 produce identical sequences.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(SEP)
    print("  All 7 tests passed. NoiseModel is ready.")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
