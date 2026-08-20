class FrequencyDiscriminator:
    """
    Converts symmetric Ramsey excitation measurements
    into a frequency error signal.
    """

    def __init__(self, probe_offset=2.5):
        self.probe_offset = probe_offset

    def error_signal(self, probability_plus, probability_minus):
        """
        Positive error  -> laser is above resonance.
        Negative error  -> laser is below resonance.
        """
        return probability_minus - probability_plus

    def estimate_detuning(
        self,
        probability_plus,
        probability_minus,
    ):
        """
        Estimate the laser detuning from the
        normalized Ramsey discriminator signal.
        """

        error = self.error_signal(
            probability_plus,
            probability_minus,
        )

        return error * self.probe_offset

    def reset(self):
        self.probe_offset = 2.5

    def __repr__(self):
        return (
            f"<FrequencyDiscriminator "
            f"probe_offset={self.probe_offset} Hz>"
        )