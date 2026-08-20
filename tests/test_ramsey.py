from simulator.ramsey import RamseyInterrogation


ramsey = RamseyInterrogation(
    interrogation_time=0.1
)

print("Interrogation Time :", ramsey.interrogation_time)

# Exactly on resonance
p0 = ramsey.excitation_probability(0.0)

print("P(detuning=0)      :", p0)

# 2.5 Hz detuning -> phase = pi/2
p1 = ramsey.excitation_probability(2.5)

print("P(detuning=2.5 Hz) :", p1)

# 5 Hz detuning -> phase = pi
p2 = ramsey.excitation_probability(5.0)

print("P(detuning=5 Hz)   :", p2)

assert abs(p0 - 1.0) < 1e-12
assert abs(p1 - 0.5) < 1e-12
assert abs(p2 - 0.0) < 1e-12

assert 0.0 <= p0 <= 1.0
assert 0.0 <= p1 <= 1.0
assert 0.0 <= p2 <= 1.0

print("\n[PASS] Ramsey interrogation tests passed.")
print(ramsey)