from controller.servo import ServoController

servo = ServoController()

errors = [
    0.20,
    0.15,
    0.10,
    0.05,
    0.02,
    0.00,
    -0.02,
    -0.05,
]

for error in errors:

    correction = servo.update(error)

    print(
        f"Error: {error:+.4f} Hz"
        f"    Correction: {correction:+.6f}"
    )
