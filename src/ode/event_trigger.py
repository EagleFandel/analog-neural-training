from __future__ import annotations

from typing import Callable, Tuple
import numpy as np


def event_triggered_fixed_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float,
    cache: dict,
    y_tol_rel: float = 1e-3,
):
    """Heuristic event-triggered evaluation: reuse last f if state change is tiny.

    If ||y - y_cached|| / (||y||+1e-12) < y_tol_rel, reuse cached k = f(t_cached,y_cached)
    to compute an Euler step; otherwise recompute f and refresh cache.
    """
    y_cached = cache.get("y")
    k_cached = cache.get("k")
    if y_cached is not None and k_cached is not None:
        denom = max(1e-12, float(np.linalg.norm(y)))
        rel = float(np.linalg.norm(y - y_cached) / denom)
        if rel < y_tol_rel:
            return y + dt * k_cached, cache

    k = f(t, y)
    cache = {"y": y.copy(), "k": k.copy(), "t": float(t)}
    return y + dt * k, cache



