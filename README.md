# ECON 743 - PS1 (Python)

This project is a Python implementation of the sovereign default model in `PS1_v1.pdf` (Arellano-style setup).

## Project Structure

- `notebooks/ps1_main.ipynb`: main workflow notebook (run this for the assignment output).
- `src/arellano_ps1/config.py`: model and numerical parameters.
- `src/arellano_ps1/discretization.py`: Rouwenhorst discretization and asset grid construction.
- `src/arellano_ps1/solver.py`: inner loop (government problem) and outer loop (equilibrium bond pricing).
- `src/arellano_ps1/checks.py`: equilibrium consistency diagnostics.
- `src/arellano_ps1/plots.py`: figure generation helpers for assignment plots.
- `src/arellano_ps1/pipeline.py`: high-level run + figure creation functions.
- `outputs/figures/`: generated figures.
- `tests/smoke_test.py`: small-grid smoke test.

## Environment

Recommended Python version: `3.11+`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Open `notebooks/ps1_main.ipynb`.
2. Run all cells.
3. Figures are saved to `outputs/figures/`.

## Notes

- Baseline parameters and grid settings match the assignment:
  - `beta=0.95`, `rho=0.94`, `sigma_z=0.03`, `r=0.02`, `y_default=0.8`, `theta=0`
  - `n_b=400`, `n_y=7`, `b_max=50`, `nu=1`
- The notebook prints convergence status and equilibrium diagnostics.
