from simulator.clock import QuantumClock

clock = QuantumClock(seed=42)

for _ in range(10):

    result = clock.step()

    print("=" * 60)

    print(f"Time                 : {result['time']}")
    print(f"Measured Frequency   : {result['measured_frequency']}")
    print(f"Frequency Offset     : {result['frequency_offset']:.10e}")
    print(f"Fractional Frequency : {result['fractional_frequency']:.10e}")
    print(f"Servo Correction     : {result['servo_correction']:.10e}")

    print("\nNoise")

    for k, v in result["noise"].items():
        print(f"{k:15}: {v:.10e}")