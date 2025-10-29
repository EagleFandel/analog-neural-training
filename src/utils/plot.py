from __future__ import annotations

from typing import Dict, List
import matplotlib.pyplot as plt


def plot_curves(curves: Dict[str, List[float]], title: str = "Training Curves", xlabel: str = "Step", ylabel: str = "Loss") -> None:
    for name, values in curves.items():
        plt.plot(values, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



