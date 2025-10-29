from __future__ import annotations

import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import csv
import os
from pathlib import Path
import json

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.mlp import MLP
from src.utils.seed import set_global_seed
from src.ode.integrators import NFECounter, rk4_step
from src.ode.geometry import make_natural_gradient_field
from src.ode.vector_fields import make_supervised_vector_field
from src.optim.baseline_adam import adam_train


def preprocess_mnist(sample_size: int, seed: int):
    try:
        data = fetch_openml("mnist_784", version=1, as_frame=False)
        x = data.data.astype(np.float64) / 255.0
        y = data.target.astype(int)
    except Exception:
        # Offline fallback: sklearn digits (8x8)
        d = load_digits()
        x = d.data.astype(np.float64) / 16.0
        y = d.target.astype(int)
        # limit sample_size to available
        sample_size = min(sample_size, x.shape[0])
    # 如果 sample_size 是绝对数量且超过数据量，则将其转换为比例 0.8
    if isinstance(sample_size, int) and sample_size >= x.shape[0]:
        train_size = 0.8
    else:
        train_size = sample_size
        if isinstance(train_size, int):
            # 限制上界
            train_size = min(train_size, x.shape[0] - 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, stratify=y, random_state=seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def run(sample_size: int, steps: int, dt: float, seed: int, out_csv: str) -> None:
    set_global_seed(seed)
    x_tr, x_te, y_tr, y_te = preprocess_mnist(sample_size, seed)

    input_dim = x_tr.shape[1]
    num_classes = int(np.max(y_tr)) + 1
    model = MLP([input_dim, 256, 128, num_classes])
    theta = model.theta0.copy()

    # Adam baseline
    theta_adam, losses_adam = adam_train(
        model.loss_and_grad,
        theta,
        x_tr,
        y_tr,
        steps=steps,
        lr=1e-3,
        task="classification",
    )
    preds_adam = model.forward(theta_adam, x_te, task="classification")
    acc_adam = np.mean(np.argmax(preds_adam, axis=1) == y_te)

    # Natural-gradient ODE with RK4
    theta_ng = theta.copy()
    f = make_natural_gradient_field(model.loss_and_grad, x_tr, y_tr, task="classification")
    nfe = NFECounter()
    losses_ng = []
    t = 0.0
    for _ in range(int(steps)):
        loss, _ = model.loss_and_grad(theta_ng, x_tr, y_tr, task="classification")
        losses_ng.append(float(loss))
        theta_ng = rk4_step(f, t, theta_ng, dt, nfe)
        t += dt
    preds_ng = model.forward(theta_ng, x_te, task="classification")
    acc_ng = np.mean(np.argmax(preds_ng, axis=1) == y_te)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "adam_loss", "ng_loss"])
        for i in range(steps):
            w.writerow([i, losses_adam[i] if i < len(losses_adam) else "", losses_ng[i] if i < len(losses_ng) else ""])

    p = Path(out_csv)
    metrics_path = p.with_name(p.stem + "_metrics.json")
    metrics = {
        "adam_acc": float(acc_adam),
        "ng_acc": float(acc_ng),
        "nfe_ng": nfe.count,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_size", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="results/mnist_ng_vs_adam.csv")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.sample_size, args.steps, args.dt, args.seed, args.out_csv)


