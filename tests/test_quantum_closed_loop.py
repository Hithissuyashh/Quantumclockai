from simulator.clock import QuantumClock


clock = QuantumClock(seed=42)

# Introduce a deliberate +0.5 Hz oscillator error
clock.oscillator.apply_noise(0.5)

print("=" * 70)
print("QUANTUM CLOSED-LOOP TEST")
print("=" * 70)

initial_offset = clock.oscillator.measure_error()

print("Initial oscillator offset :", initial_offset, "Hz")

for step in range(10):

    result = clock.step()

    print(
        f"Step {step + 1:02d} | "
        f"True detuning: {result['true_detuning']:+.6f} Hz | "
        f"Measured: {result['measured_offset']:+.6f} Hz | "
        f"Estimated: {result['estimated_offset']:+.6f} Hz | "
        f"Correction: {result['servo_correction']:+.6f}"
    )

final_offset = clock.oscillator.measure_error()

print("\nFinal oscillator offset :", final_offset, "Hz")

assert abs(final_offset) < abs(initial_offset)

print("\n[PASS] Quantum closed-loop reduced the oscillator error.")