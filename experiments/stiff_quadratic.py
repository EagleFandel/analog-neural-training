from __future__ import annotations

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import csv
import os
from typing import Tuple

import numpy as np

from src.ode.integrators import NFECounter, euler_step, rk2_step, rk4_step
from src.ode.implicit import imex_step


def quadratic(x: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray]:
    loss = 0.5 * float(x.T @ Q @ x)
    grad = Q @ x
    return loss, grad


def run(dim: int, cond: float, steps: int, dt: float, method: str, out_csv: str) -> None:
    rng = np.random.default_rng(0)
    diag = np.geomspace(1.0, cond, num=dim)
    # 构造可控谱的对称正定矩阵 Q
    U, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    Q = U @ np.diag(diag) @ U.T

    theta = rng.normal(0.0, 1e-2, size=(dim,))
    results = []

    if method == "imex":
        # IMEX: 将全部刚性项放入隐式算子，显式部分为0梯度（仅用于计算loss）
        def loss_grad_f(th, *_args):
            loss, _g_full = quadratic(th, Q)
            g_explicit = np.zeros_like(th)
            return loss, g_explicit

        for step in range(steps):
            theta, loss, stats = imex_step(
                loss_grad_f,
                theta,
                None,
                None,
                dt,
                implicit_mass=0.0,
                damping=0.0,
                implicit_op=lambda v: Q @ v,
            )
            results.append((step, loss, stats.iters, stats.residual))
    else:
        stepper = {"euler": euler_step, "rk2": rk2_step, "rk4": rk4_step}[method]
        nfe = NFECounter()

        def f(_t: float, th: np.ndarray) -> np.ndarray:
            # 显式项：-grad f(th)
            _loss, g = quadratic(th, Q)
            return -g

        for step in range(steps):
            loss, _ = quadratic(theta, Q)
            results.append((step, loss, nfe.count, 0.0))
            theta = stepper(f, step * dt, theta, dt, nfe)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        if method == "imex":
            w.writerow(["step", "loss", "cg_iters", "cg_residual"])
        else:
            w.writerow(["step", "loss", "nfe", "residual"])
        w.writerows(results)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--cond", type=float, default=1e4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--method", type=str, default="rk4", choices=["euler", "rk2", "rk4", "imex"])
    ap.add_argument("--out_csv", type=str, default="results/quadratic_{method}.csv")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_csv = args.out_csv.format(method=args.method)
    run(args.dim, args.cond, args.steps, args.dt, args.method, out_csv)


