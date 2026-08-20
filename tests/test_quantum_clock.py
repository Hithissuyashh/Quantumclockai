from simulator.clock import QuantumClock


clock = QuantumClock(seed=42)

for i in range(5):

    result = clock.step()

    print("=" * 60)
    print("Step :", i + 1)

    print(
        "Atomic frequency :",
        result["atomic_frequency"]
    )

    print(
        "True detuning    :",
        result["true_detuning"]
    )

    print(
        "P(+)             :",
        result["excitation_probability_plus"]
    )

    print(
        "P(-)             :",
        result["excitation_probability_minus"]
    )

    print(
        "Measured offset  :",
        result["measured_offset"]
    )

    print(
        "Kalman estimate  :",
        result["estimated_offset"]
    )

    print(
        "Servo correction :",
        result["servo_correction"]
    )

    assert 0.0 <= result[
        "excitation_probability_plus"
    ] <= 1.0

    assert 0.0 <= result[
        "excitation_probability_minus"
    ] <= 1.0

print("\n[PASS] Quantum clock integration test passed.")