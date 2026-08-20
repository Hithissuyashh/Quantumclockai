class AtomicReference:
    """
    Simplified atomic reference for an optical quantum clock.

    The atomic transition acts as the ideal frequency reference.
    """

    def __init__(
        self,
        transition_frequency=429_228_004_229_873.0,
        atom="Sr-87",
    ):
        self.atom = atom
        self.transition_frequency = transition_frequency

    def frequency(self):
        """Return the ideal atomic transition frequency."""
        return self.transition_frequency

    def detuning(self, laser_frequency):
        """
        Difference between laser frequency and atomic transition.

        Positive  -> laser is above atomic resonance
        Negative  -> laser is below atomic resonance
        """
        return laser_frequency - self.transition_frequency

    def fractional_detuning(self, laser_frequency):
        """Return detuning relative to the atomic transition."""
        return (
            self.detuning(laser_frequency)
            / self.transition_frequency
        )

    def reset(self):
        """Restore the nominal atomic reference."""
        self.transition_frequency = 429_228_004_229_873.0

    def __repr__(self):
        return (
            f"<AtomicReference "
            f"atom={self.atom!r}, "
            f"frequency={self.transition_frequency:.3e} Hz>"
        )