from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checks import equilibrium_diagnostics
from .config import ModelConfig
from .discretization import build_asset_grid, rouwenhorst_income_process
from .solver import EquilibriumSolution, solve_equilibrium


@dataclass
class PipelineResult:
    config: ModelConfig
    y_grid: np.ndarray
    transition: np.ndarray
    b_grid: np.ndarray
    b_min: float
    solution: EquilibriumSolution
    diagnostics: dict[str, float]


def run_model(config: ModelConfig) -> PipelineResult:
    y_grid, transition = rouwenhorst_income_process(config.rho, config.sigma_z, config.n_y)
    b_grid, b_min = build_asset_grid(config, y_grid)
    solution = solve_equilibrium(config, b_grid, y_grid, transition)
    diagnostics = equilibrium_diagnostics(solution, b_grid, y_grid, transition, config.r, config.beta)
    return PipelineResult(
        config=config,
        y_grid=y_grid,
        transition=transition,
        b_grid=b_grid,
        b_min=b_min,
        solution=solution,
        diagnostics=diagnostics,
    )


def create_assignment_figures(result: PipelineResult, output_dir: Path) -> None:
    from .plots import plot_default_heatmap, plot_policy_and_value, plot_prices_and_interest

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_prices_and_interest(
        solution=result.solution,
        b_grid=result.b_grid,
        y_grid=result.y_grid,
        output_path=output_dir / "Fig1_prices_interest.png",
    )
    plot_policy_and_value(
        solution=result.solution,
        b_grid=result.b_grid,
        y_grid=result.y_grid,
        output_path=output_dir / "Fig2_policy_value.png",
    )
    plot_default_heatmap(
        solution=result.solution,
        b_grid=result.b_grid,
        y_grid=result.y_grid,
        output_path=output_dir / "Fig3_default_probability.png",
    )
