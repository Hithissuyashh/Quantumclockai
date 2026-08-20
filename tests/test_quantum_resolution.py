from simulator.clock import QuantumClock


clock = QuantumClock(seed=42)

print("=" * 70)
print("QUANTUM FREQUENCY RESOLUTION TEST")
print("=" * 70)

# Start exactly on resonance
for i in range(20):

    result = clock.step()

    print(
        f"{i+1:02d} | "
        f"noise={result['noise']['total']:+.12e} | "
        f"true={result['true_detuning']:+.12e} | "
        f"measured={result['measured_offset']:+.12e} | "
        f"estimate={result['estimated_offset']:+.12e} | "
        f"correction={result['servo_correction']:+.12e}"
    )