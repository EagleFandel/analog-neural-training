from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np


LossGradFn = Callable[[np.ndarray, np.ndarray | None, np.ndarray | None, str], Tuple[float, np.ndarray]]


@dataclass
class LinearSolveStats:
    iters: int
    residual: float


def cg_solve(A_mv: Callable[[np.ndarray], np.ndarray], b: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, LinearSolveStats]:
    x = np.zeros_like(b)
    r = b - A_mv(x)
    p = r.copy()
    rs_old = float(r @ r)
    for it in range(1, max_iter + 1):
        Ap = A_mv(p)
        alpha = rs_old / float(p @ Ap + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol:
            return x, LinearSolveStats(iters=it, residual=np.sqrt(rs_new))
        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new
    return x, LinearSolveStats(iters=max_iter, residual=np.sqrt(rs_old))


def imex_step(
    loss_and_grad_fn: LossGradFn,
    theta: np.ndarray,
    x_batch: np.ndarray | None,
    y_batch: np.ndarray | None,
    dt: float,
    implicit_mass: float = 1.0,
    damping: float = 0.0,
    task: str = "regression",
    max_iter: int = 25,
    tol: float = 1e-6,
    implicit_op: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, float, LinearSolveStats]:
    """IMEX 半隐式一步：

    (I + dt * G) theta_{n+1} = theta_n - dt * grad_f(theta_n)

    其中 G 由 implicit_op 指定（若为空，则退化为标量隐式质量 implicit_mass）。
    """
    loss, grad_f = loss_and_grad_fn(theta, x_batch, y_batch, task)
    theta_explicit = theta - dt * grad_f

    if implicit_op is None and implicit_mass == 0.0:
        return theta_explicit, float(loss), LinearSolveStats(iters=0, residual=0.0)

    rhs = theta_explicit

    if implicit_op is None:
        def A_mv(vec: np.ndarray) -> np.ndarray:
            return vec + (dt * implicit_mass + damping) * vec
    else:
        def A_mv(vec: np.ndarray) -> np.ndarray:
            return vec + dt * implicit_op(vec) + damping * vec

    theta_new, stats = cg_solve(A_mv, rhs, max_iter=max_iter, tol=tol)
    return theta_new, float(loss), stats

 
