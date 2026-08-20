from simulator.ramsey import RamseyInterrogation
from simulator.discriminator import FrequencyDiscriminator


ramsey = RamseyInterrogation(
    interrogation_time=0.1
)

discriminator = FrequencyDiscriminator(
    probe_offset=2.5
)


# Laser ABOVE resonance
detuning = 0.5

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

print("Positive detuning :", detuning)
print("P(+)              :", p_plus)
print("P(-)              :", p_minus)
print("Error signal      :", error)
print("Estimated detuning:", estimated)

assert error > 0
assert estimated > 0


# Laser BELOW resonance
detuning = -0.5

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

print("\nNegative detuning :", detuning)
print("P(+)              :", p_plus)
print("P(-)              :", p_minus)
print("Error signal      :", error)
print("Estimated detuning:", estimated)

assert error < 0
assert estimated < 0


print("\n[PASS] Frequency discriminator tests passed.")
print(discriminator)