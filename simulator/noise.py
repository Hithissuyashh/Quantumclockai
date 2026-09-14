"""
simulator/noise.py
==================
Phase 1.2 - Modular Physics-Based Noise Engine

Every noise source can be independently enabled or disabled, which
supports ablation studies (e.g. "What happens if we remove flicker
noise?") -- a key feature for reproducible research.

Noise sources modelled
----------------------
  Source                 Type          Accumulates?
  -------------------    -----------   ------------
  White Gaussian         Stochastic    No
  Random Walk (FM)       Stochastic    Yes
  Flicker (1/f)          Stochastic    No
  Laser Phase            Stochastic    Yes
  Quantum Projection     Stochastic    No
  Temperature Shift      Systematic    No
  Zeeman Shift           Systematic    No
  Blackbody Radiation    Systematic    No

Usage
-----
    from simulator.noise import NoiseModel
    from simulator.environment import LaboratoryEnvironment

    env   = LaboratoryEnvironment(seed=0)
    noise = NoiseModel(seed=42)

    env.update()
    result = noise.total_noise(env.get_state())
    print(result["total"], result["random_walk"])

Ablation study example
----------------------
    noise.disable("flicker")
    noise.disable("laser_phase")
    result = noise.total_noise(env.get_state())   # flicker + laser excluded
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Set


# ---------------------------------------------------------------------------
# Default parameters for each noise source
# (kept as module-level constants so callers can inspect / override easily)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "white": {
        "sigma": 1e-4,
    },
    "random_walk": {
        "sigma": 1e-6,
    },
    "flicker": {
        "sigma": 5e-5,        # 1/f approximated as correlated Gaussian
        "tau":   0.9,         # correlation coefficient (0 < tau < 1)
    },
    "laser_phase": {
        "sigma": 1e-5,
    },
    "qpn": {
        "atoms": 100_000,     # number of interrogated atoms
    },
    "temperature": {
        "reference":    300.0,
        "coefficient":  2e-5,
    },
    "zeeman": {
        "reference":    50e-6,  # T
        "coefficient":  5e-3,
    },
    "blackbody": {
        "reference":    300.0,
        "coefficient":  1e-13,
    },
}

# All recognised source names
ALL_SOURCES: Set[str] = {
    "white", "random_walk", "flicker", "laser_phase",
    "qpn", "temperature", "zeeman", "blackbody",
}


class NoiseModel:
    """
    Modular, physics-based noise engine for an atomic clock simulator.

    Each noise source can be independently enabled or disabled via
    enable() / disable(), making ablation studies straightforward.

    Parameters
    ----------
    seed : int, optional
        Seed for np.random.default_rng. Gives independent, reproducible
        random streams without affecting global numpy state.
    atoms : int
        Number of interrogated atoms (sets QPN floor).
    enabled : set of str, optional
        Names of noise sources to activate. Defaults to ALL_SOURCES.
        Pass a subset to start with only specific sources active.

    Attributes
    ----------
    rng               : numpy Generator  -- seeded random stream
    random_walk_state : float            -- accumulated random-walk value
    flicker_state     : float            -- current 1/f filter state
    laser_phase       : float            -- accumulated laser phase
    _enabled          : set of str       -- currently active sources
    """

    def __init__(
        self,
        seed:    Optional[int] = None,
        atoms:   int = 100_000,
        enabled: Optional[Set[str]] = None,
    ) -> None:
        # Independent random stream (modern NumPy API, better statistics)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Mutable state for accumulating processes
        self.random_walk_state: float = 0.0
        self.flicker_state:     float = 0.0
        self.laser_phase:       float = 0.0
        self.aging_rate = 5e-10      # Hz per second

        # Override default atom count if provided
        DEFAULTS["qpn"]["atoms"] = atoms

        # Active source set
        self._enabled: Set[str] = set(enabled) if enabled is not None else set(ALL_SOURCES)
        unknown = self._enabled - ALL_SOURCES
        if unknown:
            raise ValueError(f"Unknown noise sources: {unknown}. Valid: {ALL_SOURCES}")

    # ------------------------------------------------------------------
    # Enable / disable API  (for ablation studies)
    # ------------------------------------------------------------------

    def enable(self, *sources: str) -> "NoiseModel":
        """
        Enable one or more noise sources.

        Example
        -------
        noise.enable("flicker", "laser_phase")
        """
        for src in sources:
            if src not in ALL_SOURCES:
                raise ValueError(f"Unknown source '{src}'. Valid: {ALL_SOURCES}")
            self._enabled.add(src)
        return self   # allow chaining

    def disable(self, *sources: str) -> "NoiseModel":
        """
        Disable one or more noise sources.

        Example
        -------
        noise.disable("flicker")   # ablation: remove 1/f noise
        """
        for src in sources:
            if src not in ALL_SOURCES:
                raise ValueError(f"Unknown source '{src}'. Valid: {ALL_SOURCES}")
            self._enabled.discard(src)
        return self   # allow chaining

    def active_sources(self) -> Set[str]:
        """Return the set of currently enabled noise sources."""
        return set(self._enabled)

    # ------------------------------------------------------------------
    # Individual noise sources
    # ------------------------------------------------------------------

    def white_noise(self, sigma: float = DEFAULTS["white"]["sigma"]) -> float:
        """
        White Gaussian measurement noise.

            n_w ~ N(0, sigma^2)

        Represents ADC / electronics noise. Independent sample each call.

        Parameters
        ----------
        sigma : float
            Standard deviation of the white noise.
        """
        return float(self.rng.normal(0.0, sigma))

    def random_walk(self, sigma: float = DEFAULTS["random_walk"]["sigma"]) -> float:
        """
        Random-walk frequency drift (integrated white FM noise).

            x_t = x_{t-1} + epsilon,   epsilon ~ N(0, sigma)

        Accumulates over time, mimicking slow oscillator drift.

        Parameters
        ----------
        sigma : float
            Step size standard deviation.
        """
        step = self.rng.normal(0.0, sigma)
        self.random_walk_state += step
        return float(self.random_walk_state)

    def flicker_noise(
        self,
        sigma: float = DEFAULTS["flicker"]["sigma"],
        tau:   float = DEFAULTS["flicker"]["tau"],
    ) -> float:
        """
        Approximation of 1/f (flicker) FM noise via a first-order
        autoregressive filter:

            s_t = tau * s_{t-1} + (1 - tau) * epsilon_t

        The correlation coefficient tau controls the spectral slope:
        tau -> 1 gives more low-frequency (1/f) energy.

        Parameters
        ----------
        sigma : float
            Driving noise amplitude.
        tau : float
            AR(1) memory coefficient (0 < tau < 1).
        """
        epsilon = self.rng.normal(0.0, sigma)
        self.flicker_state = tau * self.flicker_state + (1.0 - tau) * epsilon
        return float(self.flicker_state)

    def laser_phase_noise(
        self,
        sigma: float = DEFAULTS["laser_phase"]["sigma"],
        dt: float = 1.0,
    ) -> float:
        """
        Laser phase noise converted to an equivalent
        instantaneous frequency perturbation.

        Phase increment:
            dphi ~ N(0, sigma^2)

        Frequency equivalent:
            df = dphi / (2*pi*dt)

        Returns:
            Frequency perturbation in Hz.
        """

        phase_increment = self.rng.normal(
            0.0,
            sigma
        )

        self.laser_phase += phase_increment

        frequency_equivalent = (
            phase_increment /
            (2.0 * np.pi * dt)
        )

        return float(frequency_equivalent)

    def quantum_projection_noise(
        self,
        atoms: Optional[int] = None,
    ) -> float:
        """
        Quantum projection noise (QPN) from finite atom number.

        For N atoms, the measurement uncertainty scales as:

            sigma_QPN = 1 / sqrt(N)

        Increasing atom count reduces this fundamental limit.

        Parameters
        ----------
        atoms : int, optional
            Number of interrogated atoms. Defaults to self value set at init.
        """
        n = atoms if atoms is not None else int(DEFAULTS["qpn"]["atoms"])
        sigma = sigma = 3e-14
        return float(self.rng.normal(0.0, sigma))

    def temperature_shift(
        self,
        temperature:             float,
        reference_temperature:   float = DEFAULTS["temperature"]["reference"],
        coefficient:             float = DEFAULTS["temperature"]["coefficient"],
    ) -> float:
        """
        Systematic frequency shift from temperature deviation.

            Delta_f = k_T * (T - T_0)

        Parameters
        ----------
        temperature : float
            Current temperature [K].
        reference_temperature : float
            Nominal operating temperature [K].
        coefficient : float
            Temperature sensitivity [Hz / K] or fractional [1/K].
        """
        return float(coefficient * (temperature - reference_temperature))

    def zeeman_shift(
        self,
        magnetic:    float,
        reference:   float = DEFAULTS["zeeman"]["reference"],
        coefficient: float = DEFAULTS["zeeman"]["coefficient"],
    ) -> float:
        """
        Quadratic Zeeman shift from ambient magnetic field perturbation.

            Delta_f = k_B * (B - B_0)

        Parameters
        ----------
        magnetic : float
            Current magnetic field magnitude [T].
        reference : float
            Nominal field [T] (default: Earth's field ~50 µT).
        coefficient : float
            Field sensitivity coefficient.
        """
        return float(coefficient * (magnetic - reference))

    def blackbody_shift(
        self,
        temperature: float,
        reference:   float = DEFAULTS["blackbody"]["reference"],
        coefficient: float = DEFAULTS["blackbody"]["coefficient"],
    ) -> float:
        """
        Blackbody radiation (BBR) shift — a dominant systematic in optical clocks.

            Delta_f = k * (T^4 - T_ref^4)

        Parameters
        ----------
        temperature : float
            Current temperature [K].
        reference : float
            Reference temperature [K].
        coefficient : float
            BBR sensitivity coefficient.
        """
        return float(coefficient * (temperature ** 4 - reference ** 4))

    def aging_drift(self, elapsed_time):
        return self.aging_rate * elapsed_time

    # ------------------------------------------------------------------
    # Combined interface
    # ------------------------------------------------------------------

    def total_noise(
        self,
        environment: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute all active noise contributions and return per-source
        values plus their sum.

        Each disabled source contributes exactly 0.0, making ablation
        studies transparent — callers never need to change call sites.

        Parameters
        ----------
        environment : dict
            Must contain at least:
                "temperature" : float [K]
                "magnetic"    : float [T]

        Returns
        -------
        dict with keys:
            "white"        : white Gaussian noise sample
            "random_walk"  : current accumulated random-walk value
            "flicker"      : current 1/f filter output
            "laser_phase"  : current accumulated laser phase
            "qpn"          : quantum projection noise sample
            "temperature"  : temperature-induced frequency shift
            "zeeman"       : Zeeman frequency shift
            "blackbody"    : blackbody radiation frequency shift
            "aging"        : aging drift contribution
            "total"        : sum of all active contributions
        """
        T = environment["temperature"]
        B = environment["magnetic"]

        # Evaluate each source (0.0 if disabled)
        components: Dict[str, float] = {
            "white":        self.white_noise()             if "white"        in self._enabled else 0.0,
            "random_walk":  self.random_walk()             if "random_walk"  in self._enabled else 0.0,
            "flicker":      self.flicker_noise()           if "flicker"      in self._enabled else 0.0,
            "laser_phase":  self.laser_phase_noise(dt=environment.get("dt", 1.0)) if "laser_phase" in self._enabled else 0.0,
            "qpn":          self.quantum_projection_noise() if "qpn"         in self._enabled else 0.0,
            "temperature":  self.temperature_shift(T)      if "temperature"  in self._enabled else 0.0,
            "zeeman":       self.zeeman_shift(B)            if "zeeman"       in self._enabled else 0.0,
            "blackbody":    self.blackbody_shift(T)         if "blackbody"    in self._enabled else 0.0,
        }

        total = 0
        aging = self.aging_drift(environment["elapsed_time"])
        total += aging
        total += sum(components.values())

        components["aging"] = aging
        components["total"] = total
        return components

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """
        Reset all accumulated (stateful) noise processes to zero.

        Useful between independent simulation runs without re-creating
        the object (preserves the RNG stream and enabled set).
        """
        self.rng = np.random.default_rng(self.seed)
        self.random_walk_state = 0.0
        self.flicker_state     = 0.0
        self.laser_phase       = 0.0

    def state_snapshot(self) -> Dict[str, float]:
        """Return a snapshot of all internal accumulator states."""
        return {
            "random_walk_state": self.random_walk_state,
            "flicker_state":     self.flicker_state,
            "laser_phase":       self.laser_phase,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        enabled_str = ", ".join(sorted(self._enabled))
        return f"NoiseModel(enabled=[{enabled_str}])"
