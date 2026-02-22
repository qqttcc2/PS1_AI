from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig


@dataclass
class InnerSolution:
    vc: np.ndarray
    vo: np.ndarray
    policy_idx: np.ndarray
    default_indicator: np.ndarray
    iterations: int
    converged: bool


@dataclass
class EquilibriumSolution:
    q: np.ndarray
    delta: np.ndarray
    vc: np.ndarray
    vo: np.ndarray
    v_default: np.ndarray
    policy_idx: np.ndarray
    default_indicator: np.ndarray
    policy_bprime: np.ndarray
    bond_price_on_policy: np.ndarray
    interest_rate_on_policy: np.ndarray
    inner_iterations: int
    outer_iterations: int
    converged: bool


def _utility_log(consumption: np.ndarray) -> np.ndarray:
    utility = np.full_like(consumption, -np.inf, dtype=float)
    mask = consumption > 0.0
    utility[mask] = np.log(consumption[mask])
    return utility


def solve_government_given_prices(
    q: np.ndarray,
    b_grid: np.ndarray,
    y_grid: np.ndarray,
    transition: np.ndarray,
    config: ModelConfig,
    v_default: np.ndarray,
) -> InnerSolution:
    n_b = b_grid.size
    n_y = y_grid.size

    vc = np.zeros((n_b, n_y), dtype=float)
    vo = np.maximum(vc, v_default)
    policy_idx = np.zeros((n_b, n_y), dtype=int)
    row_idx = np.arange(n_b)

    b_current = b_grid[:, None]
    b_prime = b_grid[None, :]
    utility_cache = []
    for j in range(n_y):
        y_val = y_grid[j]
        repayment_matrix = y_val + b_current - (q[:, j][None, :] * b_prime)
        utility_cache.append(_utility_log(repayment_matrix))

    converged = False
    iterations = config.max_inner_iter

    for it in range(1, config.max_inner_iter + 1):
        expected_vo = vo @ transition.T
        vc_new = np.empty_like(vc)
        policy_new = np.empty_like(policy_idx)

        for j in range(n_y):
            rhs = utility_cache[j] + config.beta * expected_vo[:, j][None, :]
            policy_new[:, j] = np.argmax(rhs, axis=1)
            vc_new[:, j] = rhs[row_idx, policy_new[:, j]]

        vo_new = np.maximum(vc_new, v_default)
        diff = np.max(np.abs(vo_new - vo))

        vc = vc_new
        vo = vo_new
        policy_idx = policy_new

        if diff < config.inner_tol:
            converged = True
            iterations = it
            break

    default_indicator = (v_default >= vc).astype(float)
    return InnerSolution(
        vc=vc,
        vo=vo,
        policy_idx=policy_idx,
        default_indicator=default_indicator,
        iterations=iterations,
        converged=converged,
    )


def solve_equilibrium(
    config: ModelConfig,
    b_grid: np.ndarray,
    y_grid: np.ndarray,
    transition: np.ndarray,
) -> EquilibriumSolution:
    n_b = b_grid.size
    n_y = y_grid.size
    if transition.shape != (n_y, n_y):
        raise ValueError("transition matrix shape must be (n_y, n_y)")

    v_default = np.full((n_b, n_y), config.v_default, dtype=float)
    q = np.full((n_b, n_y), 1.0 / (1.0 + config.r), dtype=float)
    delta = np.zeros((n_b, n_y), dtype=float)

    latest_inner = None
    converged = False
    outer_iterations = config.max_outer_iter

    for outer_it in range(1, config.max_outer_iter + 1):
        inner = solve_government_given_prices(
            q=q,
            b_grid=b_grid,
            y_grid=y_grid,
            transition=transition,
            config=config,
            v_default=v_default,
        )
        latest_inner = inner

        delta_new = inner.default_indicator @ transition.T
        q_candidate = (1.0 - delta_new) / (1.0 + config.r)
        q_new = config.q_relax * q_candidate + (1.0 - config.q_relax) * q

        diff_q = np.max(np.abs(q_new - q))
        q = q_new
        delta = delta_new

        if diff_q < config.outer_tol and inner.converged:
            converged = True
            outer_iterations = outer_it
            break

    if latest_inner is None:
        raise RuntimeError("equilibrium solver failed before first iteration")

    col_idx = np.arange(n_y)[None, :]
    policy_bprime = b_grid[latest_inner.policy_idx]
    bond_price_on_policy = q[latest_inner.policy_idx, col_idx]
    interest_rate_on_policy = np.where(
        bond_price_on_policy > 1e-12,
        1.0 / bond_price_on_policy - 1.0,
        np.inf,
    )

    return EquilibriumSolution(
        q=q,
        delta=delta,
        vc=latest_inner.vc,
        vo=latest_inner.vo,
        v_default=v_default,
        policy_idx=latest_inner.policy_idx,
        default_indicator=latest_inner.default_indicator,
        policy_bprime=policy_bprime,
        bond_price_on_policy=bond_price_on_policy,
        interest_rate_on_policy=interest_rate_on_policy,
        inner_iterations=latest_inner.iterations,
        outer_iterations=outer_iterations,
        converged=converged,
    )
