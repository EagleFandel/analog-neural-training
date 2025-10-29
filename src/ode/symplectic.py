from __future__ import annotations

from typing import Tuple, Optional
import numpy as np


def damped_symplectic_heavy_ball_step(
    loss_and_grad_fn,
    w: np.ndarray,
    v: np.ndarray,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    dt: float,
    gamma: float = 0.1,
    task: str = "regression",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    r"""Quasi-symplectic step for heavy-ball ODE with damping.

    ODE:  \dot w = v,  \dot v = -gamma v - grad L(w)

    Split (Verlet-like with damping via exponential factor):
      v <- exp(-gamma*dt/2) (v - (dt/2)*grad L(w))
      w <- w + dt * v
      v <- exp(-gamma*dt/2) (v - (dt/2)*grad L(w))
    """
    loss0, g0 = loss_and_grad_fn(w, x_batch, y_batch, task)
    damp = float(np.exp(-gamma * dt * 0.5)) if gamma != 0.0 else 1.0
    v_half = damp * (v - 0.5 * dt * g0)
    w_new = w + dt * v_half
    loss1, g1 = loss_and_grad_fn(w_new, x_batch, y_batch, task)
    v_new = damp * (v_half - 0.5 * dt * g1)
    return w_new, v_new, float(loss0), float(loss1)


def integrate_heavy_ball(
    loss_and_grad_fn,
    w0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    dt: float,
    gamma: float = 0.1,
    task: str = "regression",
    v0: Optional[np.ndarray] = None,
):
    w = w0.copy()
    v = np.zeros_like(w) if v0 is None else v0.copy()
    losses = []
    energies = []
    for _ in range(int(steps)):
        w, v, l0, l1 = damped_symplectic_heavy_ball_step(
            loss_and_grad_fn, w, v, x, y, dt, gamma, task
        )
        # energy proxy: potential + kinetic
        kin = 0.5 * float(np.dot(v, v)) / v.size
        energies.append(l1 + kin)
        losses.append(l1)
    return w, v, losses, energies


