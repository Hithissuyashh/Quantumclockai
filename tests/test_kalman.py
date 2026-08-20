import numpy as np
import matplotlib.pyplot as plt

from estimator.kalman import KalmanFilter

np.random.seed(42)

true_value = 0.05

measurements = (
    true_value +
    np.random.normal(0, 0.02, 200)
)

kf = KalmanFilter()

estimates = []

for z in measurements:

    estimates.append(
        kf.update(z)
    )

plt.figure(figsize=(12,5))

plt.plot(measurements, label="Measurements")

plt.plot(estimates, linewidth=2, label="Kalman Estimate")

plt.axhline(true_value, color="red", linestyle="--", label="True Value")

plt.legend()

plt.title("Kalman Filter Validation")

plt.grid(True)

plt.show()