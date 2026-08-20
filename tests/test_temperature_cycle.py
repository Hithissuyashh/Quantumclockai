import matplotlib.pyplot as plt

from simulator.environment import LaboratoryEnvironment

env = LaboratoryEnvironment(seed=42)

# -------- TEST SETTINGS --------
# Use 100 seconds only for visualization.
# Comment this line for the real 24-hour simulation.
env.daily_period = 100.0

temperature = []
time = []

for _ in range(300):

    state = env.update()

    time.append(state["time"])
    temperature.append(state["temperature"] - 300.0)

plt.figure(figsize=(12,5))
plt.plot(time, temperature)

plt.title("Laboratory Thermal Cycle")
plt.xlabel("Time (s)")
plt.ylabel("Temperature Deviation (K)")
plt.grid(True)

plt.show()