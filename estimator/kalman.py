class KalmanFilter:
    """
    1D Kalman Filter for estimating clock frequency offset.
    """

    def __init__(
        self,
        process_variance=1e-8,
        measurement_variance=1e-8,
        initial_estimate=0.0,
        initial_error=1.0,
    ):

        # Estimated state
        self.x = initial_estimate

        # Estimation uncertainty
        self.P = initial_error

        # Process noise covariance
        self.Q = process_variance

        # Measurement noise covariance
        self.R = measurement_variance

    def update(self, measurement):
        """
        Perform one Kalman filter update.
        """

        # -------------------------
        # Prediction
        # -------------------------
        x_pred = self.x
        P_pred = self.P + self.Q

        # -------------------------
        # Kalman Gain
        # -------------------------
        K = P_pred / (P_pred + self.R)

        # -------------------------
        # Correction
        # -------------------------
        self.x = x_pred + K * (measurement - x_pred)

        self.P = (1 - K) * P_pred

        return self.x

    def reset(self):

        self.x = 0.0
        self.P = 1.0