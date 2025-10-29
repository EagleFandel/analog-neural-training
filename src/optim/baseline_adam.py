from __future__ import annotations

from typing import List, Tuple
import numpy as np


def adam_train(
    loss_and_grad_fn,
    theta0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    task: str = "regression",
) -> Tuple[np.ndarray, List[float]]:
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    losses: List[float] = []
    for t in range(1, int(steps) + 1):
        loss, g = loss_and_grad_fn(theta, x, y, task)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        losses.append(float(loss))
    return theta, losses



