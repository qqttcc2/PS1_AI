from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .solver import EquilibriumSolution


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_prices_and_interest(
    solution: EquilibriumSolution,
    b_grid: np.ndarray,
    y_grid: np.ndarray,
    output_path: Path,
) -> None:
    _ensure_parent_dir(output_path)
    y_idx_min = 0
    y_idx_max = len(y_grid) - 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(b_grid, solution.q[:, y_idx_min], linestyle="--", linewidth=1.8, label="y_min")
    axes[0].plot(b_grid, solution.q[:, y_idx_max], linewidth=1.8, label="y_max")
    axes[0].set_xlabel("B'")
    axes[0].set_ylabel("q(B', y)")
    axes[0].set_title("Bond Price Schedule")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        b_grid,
        solution.interest_rate_on_policy[:, y_idx_min],
        linestyle="--",
        linewidth=1.8,
        label="y_min",
    )
    axes[1].plot(
        b_grid,
        solution.interest_rate_on_policy[:, y_idx_max],
        linewidth=1.8,
        label="y_max",
    )
    axes[1].set_xlabel("B")
    axes[1].set_ylabel("r^C(B,y)=1/q(B'(B,y),y)-1")
    axes[1].set_title("Interest Rate Along Policy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_policy_and_value(
    solution: EquilibriumSolution,
    b_grid: np.ndarray,
    y_grid: np.ndarray,
    output_path: Path,
) -> None:
    _ensure_parent_dir(output_path)
    y_idx_min = 0
    y_idx_max = len(y_grid) - 1

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8))

    axes[0].plot(b_grid, solution.policy_bprime[:, y_idx_min], linestyle="--", linewidth=1.8, label="y_min")
    axes[0].plot(b_grid, solution.policy_bprime[:, y_idx_max], linewidth=1.8, label="y_max")
    axes[0].plot(b_grid, b_grid, linestyle="-.", linewidth=1.2, label="B'=B")
    axes[0].set_xlabel("B")
    axes[0].set_ylabel("B'(B,y)")
    axes[0].set_title("Savings Function")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(b_grid, solution.vo[:, y_idx_min], linestyle="--", linewidth=1.8, label="y_min")
    axes[1].plot(b_grid, solution.vo[:, y_idx_max], linewidth=1.8, label="y_max")
    axes[1].set_xlabel("B")
    axes[1].set_ylabel("v_o(B,y)")
    axes[1].set_title("Value Function")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_default_heatmap(
    solution: EquilibriumSolution,
    b_grid: np.ndarray,
    y_grid: np.ndarray,
    output_path: Path,
) -> None:
    _ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    heatmap = ax.imshow(
        solution.delta.T,
        aspect="auto",
        origin="lower",
        extent=[float(b_grid.min()), float(b_grid.max()), float(y_grid.min()), float(y_grid.max())],
        interpolation="nearest",
    )
    ax.set_xlabel("B'")
    ax.set_ylabel("y")
    ax.set_title("Default Probability δ(B', y)")
    fig.colorbar(heatmap, ax=ax, label="δ")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
