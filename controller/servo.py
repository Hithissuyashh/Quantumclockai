class ServoController:

    def __init__(
        self,
        kp=0.05,
        ki=0.0002,
        integral_limit=0.01,
        correction_limit=0.001,
    ):

        self.kp = kp
        self.ki = ki

        self.integral_limit = integral_limit
        self.correction_limit = correction_limit

        self.integral = 0.0

    def update(self, frequency_error):

        # Integrate error
        self.integral += frequency_error

        # Anti-windup
        self.integral = max(
            -self.integral_limit,
            min(
                self.integral,
                self.integral_limit
            )
        )

        # PI controller
        correction = (
            self.kp * frequency_error +
            self.ki * self.integral
        )

        # Limit maximum correction
        correction = max(
            -self.correction_limit,
            min(
                correction,
                self.correction_limit
            )
        )

        return correction

    def reset(self):

        self.integral = 0.0
