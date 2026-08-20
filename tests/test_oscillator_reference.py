from simulator.oscillator import OpticalOscillator
from simulator.atomic_reference import AtomicReference


atomic = AtomicReference()
oscillator = OpticalOscillator()

print("Atomic frequency     :", atomic.frequency())
print("Oscillator frequency :", oscillator.frequency())
print("Initial offset       :", oscillator.measure_error())

assert (
    oscillator.nominal_frequency
    == atomic.frequency()
)

assert (
    oscillator.frequency()
    == atomic.frequency()
)

# Apply a known frequency offset
oscillator.apply_noise(0.5)

print("\nAfter +0.5 Hz offset")
print("Oscillator frequency :", oscillator.frequency())
print("Offset               :", oscillator.measure_error())

assert abs(
    oscillator.measure_error() - 0.5
) < 1e-12

assert abs(
    oscillator.frequency()
    - (atomic.frequency() + 0.5)
) < 1e-6

# Apply correction
oscillator.apply_correction(0.5)

print("\nAfter correction")
print("Oscillator frequency :", oscillator.frequency())
print("Offset               :", oscillator.measure_error())

assert abs(
    oscillator.measure_error()
) < 1e-12

print("\n[PASS] Oscillator/atomic reference consistency test passed.")
print(oscillator)