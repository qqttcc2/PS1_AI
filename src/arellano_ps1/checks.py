from __future__ import annotations

import numpy as np

from .solver import EquilibriumSolution


def equilibrium_diagnostics(
    solution: EquilibriumSolution,
    b_grid: np.ndarray,
    y_grid: np.ndarray,
    transition: np.ndarray,
    r: float,
    beta: float,
) -> dict[str, float]:
    n_b = b_grid.size
    n_y = y_grid.size
    col_idx = np.arange(n_y)[None, :]

    q_consistent = (1.0 - solution.delta) / (1.0 + r)
    price_error = float(np.max(np.abs(solution.q - q_consistent)))

    envelope_error = float(np.max(np.abs(solution.vo - np.maximum(solution.vc, solution.v_default))))

    expected_vo = solution.vo @ transition.T
    b_prime_idx = solution.policy_idx
    chosen_q = solution.q[b_prime_idx, col_idx]
    chosen_bprime = b_grid[b_prime_idx]
    consumption = y_grid[None, :] + b_grid[:, None] - chosen_q * chosen_bprime

    util = np.full((n_b, n_y), -np.inf, dtype=float)
    feasible = consumption > 0.0
    util[feasible] = np.log(consumption[feasible])
    bellman_policy_rhs = util + beta * expected_vo[b_prime_idx, col_idx]

    finite_mask = np.isfinite(bellman_policy_rhs) & np.isfinite(solution.vc)
    if np.any(finite_mask):
        bellman_policy_error = float(
            np.max(np.abs(solution.vc[finite_mask] - bellman_policy_rhs[finite_mask]))
        )
    else:
        bellman_policy_error = float("inf")

    return {
        "max_price_error": price_error,
        "max_envelope_error": envelope_error,
        "max_bellman_policy_error": bellman_policy_error,
        "negative_consumption_on_policy_count": float(np.sum(consumption <= 0.0)),
    }
