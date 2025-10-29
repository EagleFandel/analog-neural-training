from __future__ import annotations

from typing import Callable, Tuple
import numpy as np


def make_supervised_vector_field(
    loss_and_grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, str], Tuple[float, np.ndarray]],
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    task: str = "regression",
    weight_decay: float = 0.0,
):
    """Return f(t, theta) = -grad_theta L(theta; batch) - wd * theta (L2).

    loss_and_grad_fn: (theta, x, y, task) -> (loss, grad_theta)
    """

    def f(_t: float, theta: np.ndarray) -> np.ndarray:
        _loss, grad = loss_and_grad_fn(theta, x_batch, y_batch, task)
        if weight_decay != 0.0:
            return -grad - weight_decay * theta
        return -grad

    return f










