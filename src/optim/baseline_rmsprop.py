from __future__ import annotations

from typing import List, Tuple
import numpy as np


def rmsprop_train(
    loss_and_grad_fn,
    theta0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    lr: float = 1e-3,
    alpha: float = 0.99,
    eps: float = 1e-8,
    task: str = "regression",
) -> Tuple[np.ndarray, List[float]]:
    theta = theta0.copy()
    s = np.zeros_like(theta)
    losses: List[float] = []
    for _ in range(int(steps)):
        loss, g = loss_and_grad_fn(theta, x, y, task)
        s = alpha * s + (1.0 - alpha) * (g * g)
        theta = theta - lr * g / (np.sqrt(s) + eps)
        losses.append(float(loss))
    return theta, losses



