"""
Visualization Package for QuantumClockAI
==========================================
Plotting utilities for clock signals, Allan deviation diagrams,
Kalman state estimates, transformer attention maps, and comparison dashboards.
"""

from .plots import (
    plot_frequency_error,
    plot_phase_error,
    plot_allan_deviation,
    plot_kalman_estimates,
    plot_attention_weights,
    plot_controller_performance,
)
from .dashboard import ClockDashboard

__all__ = [
    "plot_frequency_error",
    "plot_phase_error",
    "plot_allan_deviation",
    "plot_kalman_estimates",
    "plot_attention_weights",
    "plot_controller_performance",
    "ClockDashboard",
]
