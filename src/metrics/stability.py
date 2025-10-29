from __future__ import annotations

from typing import List, Tuple
import numpy as np


def oscillation_amplitude(series: List[float], window: int = 20) -> float:
    if len(series) < max(2, window):
        return float(np.max(series) - np.min(series)) if series else 0.0
    arr = np.asarray(series[-window:])
    return float(np.max(arr) - np.min(arr))


def energy_drift(energies: List[float]) -> float:
    if not energies:
        return 0.0
    return float(energies[-1] - energies[0])


def relative_energy_drift(energies: List[float]) -> float:
    if not energies:
        return 0.0
    base = max(1e-12, abs(energies[0]))
    return float((energies[-1] - energies[0]) / base)



