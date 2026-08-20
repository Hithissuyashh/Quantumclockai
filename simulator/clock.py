from simulator.environment import LaboratoryEnvironment
from simulator.noise import NoiseModel
from simulator.oscillator import OpticalOscillator
from simulator.atomic_reference import AtomicReference
from simulator.ramsey import RamseyInterrogation
from simulator.discriminator import FrequencyDiscriminator

from controller.servo import ServoController
from estimator.kalman import KalmanFilter

from ml.inference.ai_predictor import AIPredictor

AI_GAIN = 0.05
AI_CORRECTION_LIMIT = 0.0002


class QuantumClock:

    def __init__(self, seed=42):

        self.environment = LaboratoryEnvironment(
            seed=seed
        )

        self.noise = NoiseModel(
            seed=seed
        )

        self.oscillator = OpticalOscillator()

        # --------------------------------------------------
        # Quantum reference
        # --------------------------------------------------

        self.atomic_reference = AtomicReference()

        # --------------------------------------------------
        # Ramsey interrogation
        # --------------------------------------------------

        self.ramsey = RamseyInterrogation(
            interrogation_time=0.1
        )

        # --------------------------------------------------
        # Frequency discriminator
        # --------------------------------------------------

        self.discriminator = FrequencyDiscriminator(
            probe_offset=2.5
        )

        # --------------------------------------------------
        # Classical estimation/control
        # --------------------------------------------------

        self.kalman = KalmanFilter()

        self.servo = ServoController()

        # --------------------------------------------------
        # Transformer AI
        # --------------------------------------------------

        self.ai_predictor = AIPredictor()

    def step(self):

        # ==================================================
        # 1. Laboratory environment
        # ==================================================

        env = self.environment.update()

        # ==================================================
        # 2. Physical noise
        # ==================================================

        noise = self.noise.total_noise(
            env
        )

        # ==================================================
        # 3. Apply physical noise to oscillator
        # ==================================================

        self.oscillator.apply_noise(
            noise["total"]
        )

        # ==================================================
        # 4. Oscillator frequency
        # ==================================================

        measured_frequency = (
            self.oscillator.frequency()
        )

        # ==================================================
        # 5. Quantum reference
        # ==================================================

        atomic_frequency = (
            self.atomic_reference.frequency()
        )

        # ==================================================
        # 6. True laser detuning
        # ==================================================

        true_detuning = (
            self.oscillator.measure_error()
        )

        # ==================================================
        # 7. Ramsey interrogation
        # ==================================================

        plus_probability = (
            self.ramsey.excitation_probability(
                true_detuning
                + self.discriminator.probe_offset
            )
        )

        minus_probability = (
            self.ramsey.excitation_probability(
                true_detuning
                - self.discriminator.probe_offset
            )
        )

        # ==================================================
        # 8. Quantum frequency discriminator
        # ==================================================

        measured_offset = (
            self.discriminator.estimate_detuning(
                plus_probability,
                minus_probability,
            )
        )

        # ==================================================
        # 9. Kalman estimation
        # ==================================================

        estimated_offset = (
            self.kalman.update(
                measured_offset
            )
        )

        # ==================================================
        # 10. Existing PI servo
        # ==================================================

        correction = (
            self.servo.update(
                estimated_offset
            )
        )

        # ==================================================
        # 11. Build telemetry (before correction)
        # ==================================================

        telemetry = {

            "time":
                env["time"],

            "environment":
                env,

            "noise":
                noise,

            "atomic_frequency":
                atomic_frequency,

            "true_detuning":
                true_detuning,

            "excitation_probability_plus":
                plus_probability,

            "excitation_probability_minus":
                minus_probability,

            "measured_frequency":
                measured_frequency,

            "measured_offset":
                measured_offset,

            "estimated_offset":
                estimated_offset,

            "servo_correction":
                correction,
        }

        # ==================================================
        # 12. Transformer prediction
        # ==================================================

        ai_prediction = (
            self.ai_predictor.update(
                telemetry
            )
        )

        # ==================================================
        # 13. Validated AI extra correction
        # ==================================================

        ai_extra_correction = 0.0

        if ai_prediction is not None:

            ai_extra_correction = (
                AI_GAIN * ai_prediction
            )

            ai_extra_correction = max(
                -AI_CORRECTION_LIMIT,
                min(
                    ai_extra_correction,
                    AI_CORRECTION_LIMIT
                )
            )

        # ==================================================
        # 14. Apply BOTH corrections
        #
        # PI correction remains unchanged.
        # AI correction is an independent extra correction.
        # ==================================================

        self.oscillator.apply_correction(
            correction
        )

        if ai_prediction is not None:

            self.oscillator.apply_correction(
                ai_extra_correction
            )

        # ==================================================
        # 15. Measure corrected oscillator
        # ==================================================

        corrected_offset = (
            self.oscillator.measure_error()
        )

        # ==================================================
        # 16. Final telemetry
        # ==================================================

        telemetry.update({

            "corrected_offset":
                corrected_offset,

            "fractional_frequency":
                measured_offset /
                self.oscillator.nominal_frequency,

            "ai": {

                "prediction":
                    ai_prediction,

                "extra_correction":
                    ai_extra_correction,

                "gain":
                    AI_GAIN,

                "correction_limit":
                    AI_CORRECTION_LIMIT,

                "ready":
                    self.ai_predictor.ready,

                "history_length":
                    self.ai_predictor.history_length,

                "prediction_count":
                    self.ai_predictor.prediction_count,
            },
        })

        return telemetry

    def reset(self):

        # --------------------------------------------------
        # Reset physical system
        # --------------------------------------------------

        self.environment.reset()

        self.oscillator.reset()

        self.atomic_reference.reset()

        self.ramsey.reset()

        self.discriminator.reset()

        # --------------------------------------------------
        # Reset classical controller
        # --------------------------------------------------

        self.servo.reset()

        self.kalman.reset()

        # --------------------------------------------------
        # Reset AI history
        # --------------------------------------------------

        self.ai_predictor.reset()