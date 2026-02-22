from __future__ import annotations

import numpy as np

from .config import ModelConfig


def rouwenhorst_income_process(
    rho: float,
    sigma_eps: float,
    n_states: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_states < 2:
        raise ValueError("n_states must be >= 2")

    p = (1.0 + rho) / 2.0
    q = p
    transition = np.array([[p, 1.0 - p], [1.0 - q, q]], dtype=float)

    for n in range(3, n_states + 1):
        prev = transition
        transition = np.zeros((n, n), dtype=float)
        transition[:-1, :-1] += p * prev
        transition[:-1, 1:] += (1.0 - p) * prev
        transition[1:, :-1] += (1.0 - q) * prev
        transition[1:, 1:] += q * prev
        transition[1:-1, :] *= 0.5

    sigma_z_stationary = sigma_eps / np.sqrt(1.0 - rho**2)
    psi = sigma_z_stationary * np.sqrt(n_states - 1)
    z_grid = np.linspace(-psi, psi, n_states)
    y_grid = np.exp(z_grid)
    return y_grid, transition


def build_asset_grid(config: ModelConfig, y_grid: np.ndarray) -> tuple[np.ndarray, float]:
    y_min = float(np.min(y_grid))
    b_min = -((1.0 + config.r) * y_min) / config.r

    k = np.arange(config.n_b, dtype=float)
    b_grid = b_min + (config.b_max - b_min) * (k / (config.n_b - 1.0)) ** config.nu
    return b_grid, b_min
