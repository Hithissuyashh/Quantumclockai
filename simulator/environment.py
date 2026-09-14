import math
import numpy as np


class LaboratoryEnvironment:

    def __init__(self, seed=None, dt=1.0):
        import math

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.dt = dt
        self.daily_temperature_amplitude = 0.2   # Kelvin
        self.daily_period = 86400.0              # 24 hours

        self.temperature = 300.0
        self.magnetic = 50e-6
        self.laser_power = 1.0
        self.pressure = 101325.0
        self.humidity = 45.0

        self.time = 0
        self.history = []

    def _update_variable(self, value, target, alpha, sigma):

        noise = self.rng.normal(0, sigma)

        return value + alpha * (target - value) + noise

    def daily_temperature_cycle(self):

        return (
            self.daily_temperature_amplitude *
            math.sin(
                2 * math.pi * self.time / self.daily_period
            )
        )

    def update(self):

        # Daily thermal cycle
        thermal_cycle = self.daily_temperature_cycle()

        # Small laboratory fluctuations
        random_noise = self.rng.normal(0, 0.01)

        # Laboratory temperature
        self.temperature = (
            300.0 +
            thermal_cycle +
            random_noise
        )

        self.magnetic = self._update_variable(
            self.magnetic, 50e-6, 0.03, 0.1e-6
        )

        self.laser_power = self._update_variable(
            self.laser_power, 1.0, 0.05, 5e-4
        )

        self.pressure = self._update_variable(
            self.pressure, 101325.0, 0.02, 0.5
        )

        self.humidity = self._update_variable(
            self.humidity, 45.0, 0.02, 0.05
        )

        self.time += self.dt

        state = self.get_state()

        self.history.append(state)

        return state

    def get_state(self):

        return {
            "time": self.time,
            "dt": self.dt,
            "elapsed_time": self.time,
            "temperature": self.temperature,
            "magnetic": self.magnetic,
            "laser_power": self.laser_power,
            "pressure": self.pressure,
            "humidity": self.humidity
        }

    def reset(self):
        """Reset dynamic state while preserving the configured environment."""

        self.rng = np.random.default_rng(self.seed)

        self.temperature = 300.0
        self.magnetic = 50e-6
        self.laser_power = 1.0
        self.pressure = 101325.0
        self.humidity = 45.0

        self.time = 0.0
        self.history = []