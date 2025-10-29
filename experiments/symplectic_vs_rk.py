from __future__ import annotations

import os
import sys
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
from src.ode.integrators import NFECounter, rk4_step
from src.ode.vector_fields import make_supervised_vector_field
from src.ode.symplectic import integrate_heavy_ball
from src.metrics.stability import relative_energy_drift, oscillation_amplitude


def run(steps: int, dt: float, gamma: float, hidden: int, seed: int, out_csv: str) -> None:
    set_global_seed(seed)
    x, y = generate_sine_regression(n_samples=2048, noise_std=0.05, seed=seed)
    (x_tr, y_tr), _ = train_val_split(x, y, val_ratio=0.2, seed=seed)

    model = MLP([1, hidden, 1])

    # RK4 baseline on theta ODE
    theta_rk = model.theta0.copy()
    f = make_supervised_vector_field(model.loss_and_grad, x_tr, y_tr, task="regression")
    nfe = NFECounter()
    losses_rk: List[float] = []
    t = 0.0
    for _ in range(int(steps)):
        loss, _ = model.loss_and_grad(theta_rk, x_tr, y_tr, task="regression")
        losses_rk.append(float(loss))
        theta_rk = rk4_step(f, t, theta_rk, dt, nfe)
        t += dt

    # Symplectic heavy-ball
    theta_hb = model.theta0.copy()
    _w, _v, losses_hb, energies_hb = integrate_heavy_ball(
        model.loss_and_grad, theta_hb, x_tr, y_tr, steps=steps, dt=dt, gamma=gamma, task="regression"
    )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "rk4_loss", "hb_loss", "hb_energy"])
        for i in range(steps):
            hb_e = energies_hb[i] if i < len(energies_hb) else ""
            rk_l = losses_rk[i] if i < len(losses_rk) else ""
            hb_l = losses_hb[i] if i < len(losses_hb) else ""
            w.writerow([i, rk_l, hb_l, hb_e])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="results/symplectic_vs_rk.csv")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.steps, args.dt, args.gamma, args.hidden, args.seed, args.out_csv)


