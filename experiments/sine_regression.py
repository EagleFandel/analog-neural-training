from __future__ import annotations

import os
import sys
# ensure project root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import csv
import os
from typing import List

import numpy as np

from src.models.mlp import MLP
from src.utils.seed import set_global_seed
from src.utils.data import generate_sine_regression, train_val_split
from src.ode.integrators import NFECounter, euler_step, rk2_step, rk4_step
from src.ode.vector_fields import make_supervised_vector_field
from src.optim.baseline_gd import gd_train
from src.optim.baseline_adam import adam_train


def run(method: str, steps: int, lr: float, dt: float, hidden: int, seed: int, out_csv: str) -> None:
    set_global_seed(seed)
    x, y = generate_sine_regression(n_samples=2048, noise_std=0.05, seed=seed)
    (x_tr, y_tr), (x_va, y_va) = train_val_split(x, y, val_ratio=0.2, seed=seed)

    model = MLP([1, hidden, 1])
    theta = model.theta0.copy()

    def loss_only(th):
        l, _ = model.loss_and_grad(th, x_va, y_va, task="regression")
        return l

    losses: List[float] = []

    if method in {"gd", "adam"}:
        if method == "gd":
            theta, losses = gd_train(model.loss_and_grad, theta, x_tr, y_tr, steps=steps, lr=lr, task="regression")
        else:
            theta, losses = adam_train(model.loss_and_grad, theta, x_tr, y_tr, steps=steps, lr=lr, task="regression")
    else:
        # ODE-inspired: fixed-step integrators on theta dynamics
        # f(t, theta) = -grad L(theta)
        nfe = NFECounter()
        f = make_supervised_vector_field(model.loss_and_grad, x_tr, y_tr, task="regression", weight_decay=0.0)
        stepper = {"euler": euler_step, "rk2": rk2_step, "rk4": rk4_step}[method]
        t = 0.0
        for _ in range(int(steps)):
            loss, _ = model.loss_and_grad(theta, x_tr, y_tr, task="regression")
            losses.append(float(loss))
            theta = stepper(f, t, theta, dt, nfe)
            t += dt

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        for i, l in enumerate(losses):
            w.writerow([i, l])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="gd", choices=["gd", "adam", "euler", "rk2", "rk4"]) 
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="results/sine_{}.csv")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_csv = args.out_csv.format(args.method)
    run(args.method, args.steps, args.lr, args.dt, args.hidden, args.seed, out_csv)


