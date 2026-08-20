import matplotlib.pyplot as plt

from simulator.clock import QuantumClock


clock = QuantumClock(seed=42)

time = []

true_offset = []
measured_offset = []
estimated_offset = []

servo = []
noise = []

for _ in range(1000):

    result = clock.step()

    time.append(result["time"])

    true_offset.append(
        result["true_offset"]
    )

    measured_offset.append(
        result["measured_offset"]
    )

    estimated_offset.append(
        result["estimated_offset"]
    )

    servo.append(
        result["servo_correction"]
    )

    noise.append(
        result["noise"]["total"]
    )

# ==========================================================
# Figure 1
# ==========================================================

plt.figure(figsize=(13,5))

plt.plot(
    time,
    true_offset,
    linewidth=2,
    label="True Offset"
)

plt.plot(
    time,
    measured_offset,
    alpha=0.45,
    label="Measured Offset"
)

plt.plot(
    time,
    estimated_offset,
    linewidth=2,
    label="Kalman Estimate"
)

plt.title("True State vs Measurement vs Kalman Estimate")

plt.xlabel("Time (s)")
plt.ylabel("Frequency Offset (Hz)")

plt.grid(True)
plt.legend()

# ==========================================================
# Figure 2
# ==========================================================

plt.figure(figsize=(13,5))

plt.plot(
    time,
    servo,
    color="green"
)

plt.title("Servo Controller Output")

plt.xlabel("Time (s)")
plt.ylabel("Correction (Hz)")

plt.grid(True)

# ==========================================================
# Figure 3
# ==========================================================

plt.figure(figsize=(13,5))

plt.plot(
    time,
    noise,
    color="red"
)

plt.title("Total Physical Noise")

plt.xlabel("Time (s)")
plt.ylabel("Noise")

plt.grid(True)

plt.show()