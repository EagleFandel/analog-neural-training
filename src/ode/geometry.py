from __future__ import annotations

from typing import Callable, Tuple
import numpy as np


def make_natural_gradient_field(
    loss_and_grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, str], Tuple[float, np.ndarray]],
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    task: str = "regression",
    alpha: float = 0.99,
    eps: float = 1e-8,
):
    """Diagonal Fisher/RMSProp-like preconditioning for ODE field."""
    state = {"ema": None}

    def f(_t: float, theta: np.ndarray) -> np.ndarray:
        _loss, g = loss_and_grad_fn(theta, x_batch, y_batch, task)
        if state["ema"] is None:
            state["ema"] = g * g
        else:
            state["ema"] = alpha * state["ema"] + (1.0 - alpha) * (g * g)
        pre = 1.0 / (np.sqrt(state["ema"]) + eps)
        return -pre * g

    return f


def make_mirror_descent_field(
    loss_and_grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, str], Tuple[float, np.ndarray]],
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    task: str = "regression",
    mirror: str = "entropy",
    eps: float = 1e-8,
):
    """Mirror descent ODE: d/dt ∇ψ(θ) = -∇L(θ).

    支持：
      - entropy: ψ(θ)=∑ θ_i log θ_i - θ_i，需要 θ>0
      - euclid:  ψ(θ)=1/2 ||θ||^2
    """

    if mirror not in {"entropy", "euclid"}:
        raise ValueError("mirror must be 'entropy' or 'euclid'")

    def grad_psi(theta: np.ndarray) -> np.ndarray:
        if mirror == "euclid":
            return theta
        theta_pos = np.maximum(theta, eps)
        return np.log(theta_pos) + 1.0

    def inv_grad_psi(z: np.ndarray) -> np.ndarray:
        if mirror == "euclid":
            return z
        return np.exp(z - 1.0)

    def f(_t: float, theta: np.ndarray) -> np.ndarray:
        _loss, g = loss_and_grad_fn(theta, x_batch, y_batch, task)
        z = grad_psi(theta)
        dz_dt = -g
        z_next = z + dz_dt * 1.0  # dt handled by integrator
        theta_flow = inv_grad_psi(z_next)
        return theta_flow - theta

    return f


