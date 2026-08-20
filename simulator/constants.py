"""
simulator/constants.py
======================
Physical and simulation constants for the QuantumClockAI simulator.
"""

# ──────────────────────────────────────────────
# Fundamental Physical Constants
# ──────────────────────────────────────────────

SPEED_OF_LIGHT        = 2.99792458e8        # m/s
C                     = SPEED_OF_LIGHT      # alias

PLANCK_CONSTANT       = 6.62607015e-34      # J·s
H                     = PLANCK_CONSTANT     # alias

REDUCED_PLANCK        = 1.054571817e-34     # J·s  (ħ)
BOLTZMANN_CONSTANT    = 1.380649e-23        # J/K
KB                    = BOLTZMANN_CONSTANT  # alias

ELEMENTARY_CHARGE     = 1.602176634e-19     # C
AVOGADRO_NUMBER       = 6.02214076e23       # mol⁻¹

# ──────────────────────────────────────────────
# Cesium-133 Hyperfine Transition (SI definition of 1 second)
# ──────────────────────────────────────────────

CS133_HYPERFINE_FREQ  = 9_192_631_770.0     # Hz  (exact, defines SI second)
CS133_ATOMIC_MASS     = 2.2069e-25          # kg

# Reference optical transition frequency (Sr-87, legacy alias)
F0                    = 429_228_004_229_873.0  # Hz

# ──────────────────────────────────────────────
# Default Oscillator Parameters
# ──────────────────────────────────────────────

DEFAULT_NOMINAL_FREQ      = CS133_HYPERFINE_FREQ   # Hz
DEFAULT_Q_FACTOR          = 1e10                    # Quality factor
DEFAULT_LINEWIDTH         = DEFAULT_NOMINAL_FREQ / DEFAULT_Q_FACTOR  # Hz

# ──────────────────────────────────────────────
# Noise Model Defaults
# ──────────────────────────────────────────────

DEFAULT_WHITE_NOISE_PSD   = 1e-28           # Hz⁻¹  (white phase noise)
DEFAULT_FLICKER_NOISE_PSD = 1e-26           # Hz⁻¹  (1/f flicker noise)
DEFAULT_RANDOM_WALK_PSD   = 1e-30           # Hz⁻¹  (random walk FM)
DEFAULT_SHOT_NOISE_ATOMS  = 1e4             # Number of atoms (shot-noise limit)

# ──────────────────────────────────────────────
# Environment Defaults (legacy / systematic-shift model)
# ──────────────────────────────────────────────

DEFAULT_TEMPERATURE       = 300.0           # K  (room temperature)
T0                        = DEFAULT_TEMPERATURE  # alias

DEFAULT_MAGNETIC_FIELD    = 50e-6           # T   (Earth's field ~50 µT)
B0                        = DEFAULT_MAGNETIC_FIELD  # alias

DEFAULT_GRAVITY           = 9.80665         # m/s²
DEFAULT_PRESSURE          = 101_325.0       # Pa  (1 atm)

LASER_POWER               = 1.0             # nominal (dimensionless)

# ──────────────────────────────────────────────
# Simulation Parameters
# ──────────────────────────────────────────────

DEFAULT_DT                = 1.0             # s   (simulation timestep)
DEFAULT_DURATION          = 3600.0          # s   (1 hour default run)
DEFAULT_SEED              = 42              # RNG seed for reproducibility

# ──────────────────────────────────────────────
# Allan Deviation Reference Levels
# ──────────────────────────────────────────────

ADEV_WHITE_PM_COEFF       = 1e-14           # σ_y at τ=1 s (white PM)
ADEV_FLICKER_FM_FLOOR     = 5e-16           # σ_y flicker floor
ADEV_RANDOM_WALK_COEFF    = 1e-17           # σ_y random-walk coefficient

# ──────────────────────────────────────────────
# Dataset Generation
# ──────────────────────────────────────────────

TRAIN_RATIO               = 0.70
VAL_RATIO                 = 0.15
TEST_RATIO                = 0.15