from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergyProxy:
    """Simple energy proxy aggregator.

    E ≈ sum(FLOPs_i * e_flop)  (time-based proxies can be added separately)
    """
    e_flop: float = 1.0  # arbitrary unit scaling; use consistently across methods
    total_flops: int = 0
    total_time: float = 0.0

    def add_flops(self, flops: int) -> None:
        self.total_flops += int(flops)

    def add_time(self, seconds: float) -> None:
        self.total_time += float(seconds)

    @property
    def energy(self) -> float:
        return float(self.total_flops) * float(self.e_flop)

    @property
    def energy_time_proxy(self) -> float:
        return self.total_time * self.e_flop


