import pandas as pd

from simulator.clock import QuantumClock
from pathlib import Path

class DatasetGenerator:

    def __init__(self, seed=42):

        self.clock = QuantumClock(seed=seed)

    def generate(self, samples=1000, save_path=None):

        rows = []

        for _ in range(samples):

            result = self.clock.step()

            env = result["environment"]
            noise = result["noise"]

            rows.append({

                "time": result["time"],

                "temperature": env["temperature"],
                "magnetic": env["magnetic"],
                "laser_power": env["laser_power"],
                "pressure": env["pressure"],
                "humidity": env["humidity"],

                "frequency_offset": result["frequency_offset"],
                "fractional_frequency": result["fractional_frequency"],

                **noise
            })

        df = pd.DataFrame(rows)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)

        return df