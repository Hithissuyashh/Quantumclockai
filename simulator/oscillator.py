from simulator.atomic_reference import AtomicReference


class OpticalOscillator:

    def __init__(self):

        self.atomic_reference = AtomicReference()

        # Oscillator nominal frequency is locked to
        # the Sr-87 atomic transition.
        self.nominal_frequency = (
            self.atomic_reference.frequency()
        )

        self.offset = 0.0

    def frequency(self):

        return (
            self.nominal_frequency +
            self.offset
        )

    def measure_error(self):

        return self.offset

    def apply_noise(self, noise):

        self.offset += noise

    def apply_correction(self, correction):

        self.offset -= correction

    def reset(self):

        self.offset = 0.0

    def __repr__(self):

        return (
            f"<OpticalOscillator "
            f"frequency={self.frequency():.6f} Hz, "
            f"offset={self.offset:.6e} Hz>"
        )