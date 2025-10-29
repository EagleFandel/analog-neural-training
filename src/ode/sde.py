from __future__ import annotations

from typing import Tuple
import numpy as np


def euler_maruyama_step(
    loss_and_grad_fn,
    theta: np.ndarray,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    dt: float,
    sigma: float = 1e-3,
    task: str = "regression",
    rng: np.random.Generator | None = None,
    quant_bits: int | None = None,
) -> Tuple[np.ndarray, float]:
    """Euler–Maruyama for dθ = -∇L dt + σ dB_t with optional quantization."""
    loss, g = loss_and_grad_fn(theta, x_batch, y_batch, task)
    if rng is None:
        rng = np.random.default_rng(None)
    noise = rng.normal(0.0, 1.0, size=theta.shape).astype(theta.dtype)
    theta_new = theta - dt * g + (sigma * np.sqrt(dt)) * noise
    if quant_bits is not None:
        levels = 2 ** quant_bits
        theta_min = theta_new.min()
        theta_max = theta_new.max()
        if theta_max > theta_min:
            step = (theta_max - theta_min) / (levels - 1)
            theta_new = np.round((theta_new - theta_min) / step) * step + theta_min
    return theta_new, float(loss)


