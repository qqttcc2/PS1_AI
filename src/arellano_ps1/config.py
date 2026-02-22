from __future__ import annotations

from dataclasses import dataclass
from math import log


@dataclass(frozen=True)
class ModelConfig:
    beta: float = 0.95
    rho: float = 0.94
    sigma_z: float = 0.03
    r: float = 0.02
    y_default: float = 0.8
    theta: float = 0.0

    n_b: int = 400
    n_y: int = 7
    b_max: float = 50.0
    nu: float = 1.0

    max_inner_iter: int = 5000
    max_outer_iter: int = 5000
    inner_tol: float = 1e-7
    outer_tol: float = 1e-7
    q_relax: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must be in (0, 1)")
        if self.sigma_z <= 0.0:
            raise ValueError("sigma_z must be positive")
        if self.r <= 0.0:
            raise ValueError("r must be positive")
        if self.y_default <= 0.0:
            raise ValueError("y_default must be positive")
        if self.n_b < 2:
            raise ValueError("n_b must be at least 2")
        if self.n_y < 2:
            raise ValueError("n_y must be at least 2")
        if not (0.0 < self.q_relax <= 1.0):
            raise ValueError("q_relax must be in (0, 1]")
        if self.theta != 0.0:
            raise ValueError("This implementation assumes theta=0 (permanent autarky).")

    @property
    def v_default(self) -> float:
        return log(self.y_default) / (1.0 - self.beta)
