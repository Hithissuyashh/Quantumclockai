"""
Extended Kalman Filter (EKF) for nonlinear atomic clock models.

Extends the linear KalmanFilter to handle nonlinear dynamics such as
quadratic Zeeman shift and blackbody radiation corrections by linearizing
via first-order Taylor expansion (Jacobian).
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, List

from .filter import KalmanFilter, KalmanState
from simulator import constants as C


class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter with user-supplied nonlinear dynamics.

    Instead of a fixed transition matrix F, the EKF accepts:
      - f(x): nonlinear state transition function
      - F_jac(x): Jacobian ∂f/∂x evaluated at current state

    If not provided, falls back to the linear model of the base KalmanFilter.

    Parameters
    ----------
    f_func : Callable[[np.ndarray, float], np.ndarray], optional
        Nonlinear state transition: x_{k+1} = f(x_k, dt).
    f_jac : Callable[[np.ndarray, float], np.ndarray], optional
        Jacobian of f w.r.t. x: shape (3, 3).
    h_func : Callable[[np.ndarray], np.ndarray], optional
        Nonlinear observation function h(x). Defaults to H·x.
    h_jac : Callable[[np.ndarray], np.ndarray], optional
        Jacobian of h w.r.t. x: shape (1, 3).
    **kwargs
        Forwarded to KalmanFilter.__init__.
    """

    def __init__(
        self,
        f_func: Optional[Callable] = None,
        f_jac:  Optional[Callable] = None,
        h_func: Optional[Callable] = None,
        h_jac:  Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._f_func = f_func
        self._f_jac  = f_jac
        self._h_func = h_func
        self._h_jac  = h_jac

    # ------------------------------------------------------------------
    # Overridden predict / update
    # ------------------------------------------------------------------

    def predict(self):
        """EKF predict: propagate state through nonlinear f, linearize Q."""
        if self._f_func is not None:
            x_pred  = self._f_func(self._x, self.dt)
            F_k     = self._f_jac(self._x, self.dt) if self._f_jac else self.F
        else:
            x_pred  = self.F @ self._x
            F_k     = self.F

        self._x = x_pred
        self._P = F_k @ self._P @ F_k.T + self.Q
        return self._x.copy(), self._P.copy()

    def update(self, z: float) -> KalmanState:
        """EKF update: linearize observation around current state."""
        if self._h_func is not None:
            z_pred = self._h_func(self._x)
            H_k    = self._h_jac(self._x) if self._h_jac else self.H
        else:
            z_pred = (self.H @ self._x).ravel()
            H_k    = self.H

        inn  = np.array([[z]]) - z_pred.reshape(1, -1)
        S    = H_k @ self._P @ H_k.T + self.R
        K    = self._P @ H_k.T @ np.linalg.inv(S)

        self._x = self._x + (K @ inn.T).ravel()
        self._P = (np.eye(3) - K @ H_k) @ self._P

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

    def __repr__(self) -> str:
        nonlinear = self._f_func is not None
        return f"ExtendedKalmanFilter(dt={self.dt} s, nonlinear={nonlinear})"
