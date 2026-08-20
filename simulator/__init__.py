"""
QuantumClockAI — Simulator Package
====================================
Simulates a quantum atomic clock environment including oscillator dynamics,
environmental perturbations, and noise modeling.

Import individual modules directly to avoid circular / incomplete deps
during incremental development of each phase.
"""

from .environment import LaboratoryEnvironment
from . import constants

__all__ = [
    "LaboratoryEnvironment",
    "constants",
]

# Phase-gated imports — uncomment as each module is completed.
# from .oscillator import Oscillator
# from .noise      import NoiseModel
# from .clock      import AtomicClock
# from .dataset    import DatasetGenerator
