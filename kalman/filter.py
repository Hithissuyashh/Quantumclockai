"""
Linear Kalman Filter for atomic clock state estimation.

State vector:  x = [phase_offset (rad), freq_offset (frac), drift_rate (1/s)]
Observation:   z = total_freq_error (fractional frequency)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from simulator import constants as C


@dataclass
class KalmanState:
    """Kalman filter state snapshot."""
    time:       float
    x_hat:      np.ndarray   # (3,)  estimated state
    P:          np.ndarray   # (3,3) error covariance
    innovation: float        # measurement residual
    S:          float        # innovation covariance


class KalmanFilter:
    """
    Classical linear Kalman filter for a 3-state atomic clock model.

    State-space model
    -----------------
    State:    x = [φ,  y,  D]ᵀ
              φ — phase offset [rad]
              y — fractional frequency offset
              D — frequency drift rate [1/s]

    Dynamics: x_{k+1} = F·x_k + w_k,   w_k ~ N(0, Q)
    Measurement: z_k = H·x_k + v_k,    v_k ~ N(0, R)

    Transition matrix F (Euler, timestep T):
        [ 1   2π·f₀·T   0 ]
        [ 0      1      T ]
        [ 0      0      1 ]

    Measurement matrix H:
        [ 0   1   0 ]   (observes fractional frequency)

    Parameters
    ----------
    dt : float
        Timestep (interrogation period) [s].
    q_phase : float
        Process noise variance for phase [rad²/s].
    q_freq : float
        Process noise variance for frequency offset [1/s].
    q_drift : float
        Process noise variance for drift [1/s³].
    r_obs : float
        Measurement noise variance (fractional frequency noise).
    x0 : np.ndarray, optional
        Initial state estimate. Defaults to zeros.
    P0 : np.ndarray, optional
        Initial error covariance. Defaults to large diagonal.
    """

    def __init__(
        self,
        dt:      float = C.DEFAULT_DT,
        q_phase: float = 1e-20,
        q_freq:  float = 1e-26,
        q_drift: float = 1e-32,
        r_obs:   float = 1e-26,
        x0:      Optional[np.ndarray] = None,
        P0:      Optional[np.ndarray] = None,
    ) -> None:
        self.dt = dt
        f0 = C.DEFAULT_NOMINAL_FREQ

        # Transition matrix
        self.F = np.array([
            [1.0, 2.0 * np.pi * f0 * dt, 0.0],
            [0.0, 1.0,                   dt ],
            [0.0, 0.0,                   1.0],
        ])

        # Observation matrix (measure fractional frequency)
        self.H = np.array([[0.0, 1.0, 0.0]])

        # Process noise covariance
        self.Q = np.diag([q_phase, q_freq, q_drift])

        # Measurement noise covariance
        self.R = np.array([[r_obs]])

        # Initial state and covariance
        self._x = x0.copy() if x0 is not None else np.zeros(3)
        self._P = P0.copy() if P0 is not None else np.diag([1e-10, 1e-20, 1e-30])

        self._t: float = 0.0
        self._history: List[KalmanState] = []

    # ------------------------------------------------------------------
    # Core predict / update cycle
    # ------------------------------------------------------------------

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman predict step (time update)."""
        self._x = self.F @ self._x
        self._P = self.F @ self._P @ self.F.T + self.Q
        return self._x.copy(), self._P.copy()

    def update(self, z: float) -> KalmanState:
        """
        Kalman update step (measurement update).

        Parameters
        ----------
        z : float
            Measured fractional frequency error.

        Returns
        -------
        KalmanState
            Updated state estimate.
        """
        z_vec   = np.array([[z]])
        inn     = z_vec - self.H @ self._x       # innovation
        S       = self.H @ self._P @ self.H.T + self.R
        K       = self._P @ self.H.T @ np.linalg.inv(S)

        self._x = self._x + (K @ inn).ravel()
        self._P = (np.eye(3) - K @ self.H) @ self._P

        self._t += self.dt
        state = KalmanState(
            time=self._t,
            x_hat=self._x.copy(),
            P=self._P.copy(),
            innovation=float(inn[0, 0]),
            S=float(S[0, 0]),
        )
        self._history.append(state)
        return state

    def step(self, z: float) -> KalmanState:
        """Combined predict + update."""
        self.predict()
        return self.update(z)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def history(self) -> List[KalmanState]:
        return list(self._history)

    def reset(self, x0: Optional[np.ndarray] = None, P0: Optional[np.ndarray] = None) -> None:
        self._x = x0.copy() if x0 is not None else np.zeros(3)
        self._P = P0.copy() if P0 is not None else np.diag([1e-10, 1e-20, 1e-30])
        self._t = 0.0
        self._history.clear()

    def __repr__(self) -> str:
        return f"KalmanFilter(dt={self.dt} s, state={self._x})"
