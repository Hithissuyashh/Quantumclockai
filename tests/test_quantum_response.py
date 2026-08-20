from simulator.atomic_reference import AtomicReference
from simulator.ramsey import RamseyInterrogation
from simulator.discriminator import FrequencyDiscriminator


atomic = AtomicReference()

ramsey = RamseyInterrogation(
    interrogation_time=0.1
)

discriminator = FrequencyDiscriminator(
    probe_offset=2.5
)


def measure(detuning):

    p_plus = ramsey.excitation_probability(
        detuning + discriminator.probe_offset
    )

    p_minus = ramsey.excitation_probability(
        detuning - discriminator.probe_offset
    )

    error = discriminator.error_signal(
        p_plus,
        p_minus
    )

    estimated = discriminator.estimate_detuning(
        p_plus,
        p_minus
    )

    return p_plus, p_minus, error, estimated


print("=" * 60)
print("QUANTUM RESPONSE TEST")
print("=" * 60)


# ------------------------------------------------
# Positive detuning
# ------------------------------------------------

detuning = +0.5

p_plus, p_minus, error, estimated = measure(detuning)

print("\nPositive Detuning")
print("----------------")
print("True detuning :", detuning, "Hz")
print("P(+)          :", p_plus)
print("P(-)          :", p_minus)
print("Error signal  :", error)
print("Estimated     :", estimated, "Hz")

assert p_minus > p_plus
assert error > 0
assert estimated > 0


# ------------------------------------------------
# Negative detuning
# ------------------------------------------------

detuning = -0.5

p_plus, p_minus, error, estimated = measure(detuning)

print("\nNegative Detuning")
print("----------------")
print("True detuning :", detuning, "Hz")
print("P(+)          :", p_plus)
print("P(-)          :", p_minus)
print("Error signal  :", error)
print("Estimated     :", estimated, "Hz")

assert p_plus > p_minus
assert error < 0
assert estimated < 0


print("\n" + "=" * 60)
print("[PASS] Quantum response behaves correctly.")
print("=" * 60)