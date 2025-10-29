from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def gd_train(
    loss_and_grad_fn,
    theta0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    lr: float = 1e-2,
    task: str = "regression",
) -> Tuple[np.ndarray, List[float]]:
    """Full-batch gradient descent using model-provided loss_and_grad.

    Returns (theta, losses)
    """
    theta = theta0.copy()
    losses: List[float] = []
    for _ in range(int(steps)):
        loss, grad = loss_and_grad_fn(theta, x, y, task)
        theta = theta - float(lr) * grad
        losses.append(float(loss))
    return theta, losses



