from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arellano_ps1 import ModelConfig, run_model  # noqa: E402


def main() -> None:
    config = ModelConfig(
        n_b=80,
        n_y=5,
        max_inner_iter=800,
        max_outer_iter=500,
        inner_tol=1e-6,
        outer_tol=1e-6,
        q_relax=0.8,
    )
    result = run_model(config)
    sol = result.solution

    assert sol.q.shape == (config.n_b, config.n_y)
    assert sol.delta.shape == (config.n_b, config.n_y)
    assert sol.policy_idx.shape == (config.n_b, config.n_y)
    assert np.all(np.isfinite(sol.vo))
    assert np.all(sol.q >= 0.0)
    assert np.max(np.abs(sol.q - (1.0 - sol.delta) / (1.0 + config.r))) < 1e-9

    print("smoke_test_passed")
    print(f"converged={sol.converged}, outer_iterations={sol.outer_iterations}, inner_iterations={sol.inner_iterations}")
    print(result.diagnostics)


if __name__ == "__main__":
    main()
