from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np


RHSFunc = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class NFECounter:
    count: int = 0

    def inc(self, n: int = 1) -> None:
        self.count += n


def euler_step(f: RHSFunc, t: float, y: np.ndarray, dt: float, nfe: Optional[NFECounter] = None) -> np.ndarray:
    k1 = f(t, y)
    if nfe is not None:
        nfe.inc(1)
    return y + dt * k1


def rk2_step(f: RHSFunc, t: float, y: np.ndarray, dt: float, nfe: Optional[NFECounter] = None) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + dt, y + dt * k1)
    if nfe is not None:
        nfe.inc(2)
    return y + 0.5 * dt * (k1 + k2)


def rk4_step(f: RHSFunc, t: float, y: np.ndarray, dt: float, nfe: Optional[NFECounter] = None) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    if nfe is not None:
        nfe.inc(4)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def dopri54_step(
    f: RHSFunc,
    t: float,
    y: np.ndarray,
    dt: float,
    rtol: float = 1e-4,
    atol: float = 1e-7,
    nfe: Optional[NFECounter] = None,
) -> Tuple[np.ndarray, float, float, bool]:
    """Single adaptive Dormand–Prince(5,4) step with error control.

    Returns: (y_new, t_new, dt_suggested, accepted)
    """
    # Coefficients (Dormand–Prince tableau)
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1.0
    c7 = 1.0

    a21 = 1/5
    a31 = 3/40; a32 = 9/40
    a41 = 44/45; a42 = -56/15; a43 = 32/9
    a51 = 19372/6561; a52 = -25360/2187; a53 = 64448/6561; a54 = -212/729
    a61 = 9017/3168; a62 = -355/33; a63 = 46732/5247; a64 = 49/176; a65 = -5103/18656
    a71 = 35/384; a72 = 0; a73 = 500/1113; a74 = 125/192; a75 = -2187/6784; a76 = 11/84

    # 5th order solution weights
    b1 = 35/384; b2 = 0; b3 = 500/1113; b4 = 125/192; b5 = -2187/6784; b6 = 11/84; b7 = 0
    # 4th order solution weights
    b1s = 5179/57600; b2s = 0; b3s = 7571/16695; b4s = 393/640; b5s = -92097/339200; b6s = 187/2100; b7s = 1/40

    k1 = f(t, y)
    k2 = f(t + c2 * dt, y + dt * (a21 * k1))
    k3 = f(t + c3 * dt, y + dt * (a31 * k1 + a32 * k2))
    k4 = f(t + c4 * dt, y + dt * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = f(t + c5 * dt, y + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = f(t + c6 * dt, y + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
    y5 = y + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    k7 = f(t + c7 * dt, y5)
    y4 = y + dt * (b1s * k1 + b2s * k2 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6 + b7s * k7)
    if nfe is not None:
        nfe.inc(7)

    # Error estimate
    err = np.linalg.norm(y5 - y4, ord=np.inf)
    tol = atol + rtol * max(np.linalg.norm(y, ord=np.inf), np.linalg.norm(y5, ord=np.inf))
    safety = 0.9
    min_factor = 0.2
    max_factor = 5.0
    if err == 0.0:
        factor = max_factor
    else:
        factor = safety * (tol / err) ** 0.2
        factor = float(np.clip(factor, min_factor, max_factor))

    accept = err <= tol
    dt_new = dt * factor
    t_new = t + (dt if accept else 0.0)
    y_new = (y5 if accept else y)
    return y_new, t_new, dt_new, accept


def integrate_fixed_steps(
    f: RHSFunc,
    y0: np.ndarray,
    dt: float,
    steps: int,
    method: str = "rk4",
    nfe: Optional[NFECounter] = None,
) -> np.ndarray:
    y = y0.copy()
    t = 0.0
    for _ in range(int(steps)):
        if method == "euler":
            y = euler_step(f, t, y, dt, nfe)
        elif method == "rk2":
            y = rk2_step(f, t, y, dt, nfe)
        elif method == "rk4":
            y = rk4_step(f, t, y, dt, nfe)
        else:
            raise ValueError(f"Unknown method: {method}")
        t += dt
    return y


def integrate_adaptive(
    f: RHSFunc,
    y0: np.ndarray,
    t_end: float,
    dt_init: float = 1e-2,
    rtol: float = 1e-4,
    atol: float = 1e-7,
    nfe: Optional[NFECounter] = None,
) -> Tuple[np.ndarray, float]:
    """Integrate using adaptive Dormand–Prince until t reaches t_end."""
    y = y0.copy()
    t = 0.0
    dt = dt_init
    while t < t_end:
        dt = min(dt, t_end - t)
        y_new, t_new, dt_new, accept = dopri54_step(f, t, y, dt, rtol, atol, nfe)
        if accept:
            y = y_new
            t = t_new
        dt = dt_new
    return y, t




