import numpy as np

from simulator.environment import LaboratoryEnvironment
from simulator.noise import NoiseModel


env = LaboratoryEnvironment(seed=42)
noise = NoiseModel(seed=42)

steps = 10000

history = {}

for source in noise.active_sources():
    history[source] = []

history["total"] = []

for _ in range(steps):

    state = env.update()
    result = noise.total_noise(state)

    for key in history:
        history[key].append(result[key])


print("=" * 70)
print("NOISE COMPONENT ANALYSIS")
print("=" * 70)

for key, values in history.items():

    values = np.asarray(values)

    print(
        f"{key:15s} | "
        f"RMS = {np.sqrt(np.mean(values ** 2)):.12e} Hz | "
        f"MAX = {np.max(np.abs(values)):.12e}"
    )