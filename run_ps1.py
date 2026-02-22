from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arellano_ps1 import ModelConfig, create_assignment_figures, run_model  # noqa: E402


def main() -> None:
    config = ModelConfig()
    result = run_model(config)
    create_assignment_figures(result, PROJECT_ROOT / "outputs" / "figures")

    print(f"Converged: {result.solution.converged}")
    print(f"Outer iterations: {result.solution.outer_iterations}")
    print(f"Inner iterations (last outer step): {result.solution.inner_iterations}")
    print(result.diagnostics)


if __name__ == "__main__":
    main()
