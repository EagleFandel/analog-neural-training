from __future__ import annotations

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import csv
import os

import numpy as np

from src.models.mlp import MLP
from src.utils.seed import set_global_seed
from src.utils.data import generate_sine_regression, train_val_split
from src.ode.sde import euler_maruyama_step


def run(steps: int, dt: float, sigma: float, quant_bits: int | None, seed: int, out_csv: str) -> None:
    set_global_seed(seed)
    x, y = generate_sine_regression(n_samples=4096, noise_std=0.05, seed=seed)
    (x_tr, y_tr), (x_va, y_va) = train_val_split(x, y, val_ratio=0.2, seed=seed)

    model = MLP([1, 64, 64, 1])
    theta = model.theta0.copy()
    rng = np.random.default_rng(seed)

    results = []
    for step in range(int(steps)):
        theta, loss_tr = euler_maruyama_step(
            model.loss_and_grad,
            theta,
            x_tr,
            y_tr,
            dt,
            sigma=sigma,
            task="regression",
            rng=rng,
            quant_bits=quant_bits,
        )
        loss_va, _ = model.loss_and_grad(theta, x_va, y_va, task="regression")
        results.append((step, loss_tr, float(loss_va)))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss"])
        w.writerows(results)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--quant_bits", type=int, default=-1, help="<=0 表示不量化")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_csv", type=str, default="results/sde_sigma{sigma}_q{quant}.csv")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quant = None if args.quant_bits <= 0 else args.quant_bits
    out_csv = args.out_csv.format(sigma=args.sigma, quant="none" if quant is None else quant)
    run(args.steps, args.dt, args.sigma, quant, args.seed, out_csv)


