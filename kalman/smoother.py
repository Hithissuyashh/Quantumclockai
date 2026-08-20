"""
Rauch-Tung-Striebel (RTS) Smoother for atomic clock state estimation.

Performs a backward pass over the Kalman filter history to produce
minimum-variance smoothed state estimates x_{k|N} for all k ≤ N.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from .filter import KalmanFilter, KalmanState


@dataclass
class SmootherResult:
    """RTS smoother output for one timestep."""
    time:     float
    x_smooth: np.ndarray   # (3,) smoothed state
    P_smooth: np.ndarray   # (3,3) smoothed covariance


class RTSSmoother:
    """
    Rauch-Tung-Striebel backward smoother.

    Given a completed forward Kalman filter pass (filter history),
    performs the RTS backward sweep:

        G_k   = P_{k|k} · Fᵀ · P_{k+1|k}⁻¹
        x_{k|N} = x_{k|k} + G_k · (x_{k+1|N} − x_{k+1|k})
        P_{k|N} = P_{k|k} + G_k · (P_{k+1|N} − P_{k+1|k}) · G_kᵀ

    Parameters
    ----------
    kalman_filter : KalmanFilter
        A filter that has already completed its forward pass.
    """

    def __init__(self, kalman_filter: KalmanFilter) -> None:
        self.kf = kalman_filter

    def smooth(self) -> List[SmootherResult]:
        """
        Run RTS backward sweep over filter history.

        Returns
        -------
        list[SmootherResult]
            Smoothed states in forward time order.
        """
        history = self.kf.history
        N = len(history)
        if N == 0:
            return []

        F = self.kf.F
        Q = self.kf.Q

        # Initialize smoothed arrays from filtered estimates
        x_s = [s.x_hat.copy() for s in history]
        P_s = [s.P.copy()     for s in history]

        # Backward pass
        for k in range(N - 2, -1, -1):
            P_kk   = history[k].P
            P_k1_k = F @ P_kk @ F.T + Q       # predicted covariance at k+1

            G_k    = P_kk @ F.T @ np.linalg.inv(P_k1_k)

            x_s[k] = history[k].x_hat + G_k @ (x_s[k + 1] - F @ history[k].x_hat)
            P_s[k] = P_kk + G_k @ (P_s[k + 1] - P_k1_k) @ G_k.T

        return [
            SmootherResult(
                time=history[i].time,
                x_smooth=x_s[i],
                P_smooth=P_s[i],
            )
            for i in range(N)
        ]

    def __repr__(self) -> str:
        return f"RTSSmoother(n_steps={len(self.kf.history)})"
