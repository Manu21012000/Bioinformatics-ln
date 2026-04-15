"""
Spatial infection spread (2D grid) with 3D surface plot + mean-field SIR validation via SciPy ODE.

The spatial model is illustrative; the SIR ODE uses textbook parameters for a clean trajectory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

WORKFLOW_ROOT = Path(__file__).resolve().parents[2]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from core.config import load_config
from core.models import SimulationResult
from core.validation import validate_simulation_solution, validate_time_points
from visualizations.plots_3d import plot_infection_surface


def spatial_infection_surface(
    grid: int = 48,
    steps: int = 120,
    beta: float = 0.25,
    diffusion: float = 0.08,
    seed: int = 7,
) -> np.ndarray:
    """Discrete-time SI-like spread with diffusion on a 2D grid (infected fraction)."""
    rng = np.random.default_rng(seed)
    I = np.zeros((grid, grid), dtype=float)
    S = np.ones_like(I)
    ix, iy = grid // 2, grid // 2
    I[ix, iy] = 0.05
    S[ix, iy] = 1.0 - I[ix, iy]

    def laplacian(a: np.ndarray) -> np.ndarray:
        return (
            np.roll(a, 1, axis=0)
            + np.roll(a, -1, axis=0)
            + np.roll(a, 1, axis=1)
            + np.roll(a, -1, axis=1)
            - 4.0 * a
        )

    for _ in range(steps):
        noise = rng.normal(0.0, 0.002, size=I.shape)
        force = beta * S * I + diffusion * laplacian(I)
        I = np.clip(I + force + noise, 0.0, 1.0)
        S = np.clip(1.0 - I, 0.0, 1.0)

    return I


def sir_rhs(_t: float, y: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    s, i_, r = y
    return np.array([-beta * s * i_, beta * s * i_ - gamma * i_, gamma * i_])


def run_sir(beta: float = 0.3, gamma: float = 0.1, t_end: float = 160.0):
    y0 = np.array([0.99, 0.01, 0.0])
    return solve_ivp(
        lambda t, y: sir_rhs(t, y, beta, gamma),
        (0.0, t_end),
        y0,
        t_eval=np.linspace(0.0, t_end, 161),
        rtol=1e-8,
        atol=1e-10,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Spatial infection + SIR ODE validation.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    I = spatial_infection_surface()
    plot_infection_surface(I, args.out_dir / "infection_surface.png")

    sol = run_sir()
    validate_simulation_solution(sol, cfg)
    validate_time_points(sol.t, cfg)

    sim_model = SimulationResult(
        time_points=sol.t.tolist(),
        concentrations=[sol.y[i].tolist() for i in range(sol.y.shape[0])],
    )
    print("Pydantic SimulationResult OK:", len(sim_model.time_points), "time points")

    tbl = np.column_stack([sol.t, sol.y.T])
    hdr = "t,S,I,R"
    np.savetxt(args.out_dir / "sir_S.csv", tbl, delimiter=",", header=hdr, comments="")
    print(f"Wrote spatial surface and {args.out_dir / 'sir_S.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
