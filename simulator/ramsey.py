import math


class RamseyInterrogation:
    """
    Simplified Ramsey spectroscopy model.

    Two coherent interactions separated by a free-evolution
    interval produce an excitation probability dependent
    on laser detuning.
    """

    def __init__(
        self,
        interrogation_time=0.1,
        contrast=1.0,
    ):
        self.interrogation_time = interrogation_time
        self.contrast = contrast

    def excitation_probability(self, detuning):
        """
        Calculate excitation probability from laser detuning.

        P = 0.5 * [1 + C * cos(2*pi*detuning*T)]
        """

        phase = (
            2.0
            * math.pi
            * detuning
            * self.interrogation_time
        )

        probability = 0.5 * (
            1.0
            + self.contrast * math.cos(phase)
        )

        return probability

    def phase(self, detuning):
        """Return accumulated Ramsey phase in radians."""

        return (
            2.0
            * math.pi
            * detuning
            * self.interrogation_time
        )

    def reset(self):
        self.interrogation_time = 0.1
        self.contrast = 1.0

    def __repr__(self):
        return (
            f"<RamseyInterrogation "
            f"T={self.interrogation_time}s, "
            f"contrast={self.contrast}>"
        )