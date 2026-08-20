from simulator.atomic_reference import AtomicReference


atom = AtomicReference()

print("Atom :", atom.atom)
print("Frequency :", atom.frequency(), "Hz")

laser = atom.frequency() + 1000.0

print("Laser Frequency :", laser, "Hz")
print("Detuning :", atom.detuning(laser), "Hz")
print("Fractional Detuning :", atom.fractional_detuning(laser))

assert atom.frequency() > 4e14
assert atom.detuning(laser) == 1000.0

atom.reset()

assert atom.frequency() == 429_228_004_229_873.0

print("\n[PASS] Atomic reference tests passed.")
print(atom)