"""
Convergence studies for pure shear and AXE cases against analytical solutions.

Produces two plots:
  1. grid_convergence_all.svg — convergence of the 2nd-order phi tensor with
     number of time steps, at fixed spherical design degree.
  2. grid_convergence_M6_all.svg — convergence of the 6th-order M tensor with
     spherical design degree, at fixed number of time steps.

Outputs:
  results/convergence_plots/grid_convergence_all.svg
  results/convergence_plots/grid_convergence_M6_all.svg
"""

import argparse
import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

from src.analytical_solutions import analytical_solution_pure_shear, phi_analytical_strain
from src.rdt_solver import simulate_single
from src.structure_tensors import evaluate_M_6th_order_es
from src.tensor_utils import strain_rate

plt.style.use("seaborn-v0_8-poster")

logger = logging.getLogger(__name__)

OMEGA = np.zeros(3)
SOLVER = "rk4"
ST_MAX = 2

# phi convergence: sweep number of time steps at fixed spherical design degree
TIME_STEPS_ARR = [16, 32, 64, 128]
SD_DEGREE = 109

# M6 convergence: sweep spherical design degree at fixed number of time steps
SD_DEGREE_ARR = [11, 51, 91, 131, 171, 211, 251, 291]
SD_DEGREE_TRUE = 325
TIME_STEPS = 128

_SQ = 1 / 2**0.5
DIAG_TRANSFORM = np.array([[_SQ, 0, -_SQ], [0, 1, 0], [_SQ, 0, _SQ]])

# case label -> (velocity gradient, cache-file slug)
CASES = {
    "PS": (np.array([[1 / 2, 0, 1 / 2], [0, 0, 0], [-1 / 2, 0, -1 / 2]]), "shear"),
    "AXE": (np.array([[-1, 0, 0], [0, 1 / 2, 0], [0, 0, 1 / 2]]), "strainAXE"),
}

# indices of the six unique phi components in the flat solver state
PHI_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _results_dir() -> Path:
    """Resolves and creates rdt-sd/results/convergence_plots."""
    results_dir = Path(__file__).resolve().parents[1] / "results" / "convergence_plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _tmax(grad_u: np.ndarray) -> float:
    """Resolves integration time reaching strain time ST_MAX for grad_u (3, 3)."""
    return ST_MAX / strain_rate(grad_u[None, ...])[0]


def _cached(path: Path, compute: Callable[[], np.ndarray]) -> np.ndarray:
    """Loads array from path, or computes, caches, and returns it."""
    if path.exists():
        return np.load(path)
    array = compute()
    np.save(path, array)
    return array


def _numerical(grad_u: np.ndarray, sd_degree: int, tmax: float, time_steps: int) -> np.ndarray:
    """Runs the solver; returns its state, shape (9, num_nodes, time_steps)."""
    sol, _ = simulate_single(grad_u, tmax, time_steps, OMEGA, sd_degree, solver=SOLVER)
    return np.asarray(sol).transpose(1, 2, 0)  # (T, 9, n) -> (9, n, T)


def _analytical(case: str, sd_degree: int, tmax: float, time_steps: int) -> np.ndarray:
    """Evaluates analytical solution in solver layout, shape (9, num_nodes, time_steps)."""
    if case == "PS":
        phi, k = analytical_solution_pure_shear(
            sd_degree, tmax, time_steps, rot_matrix=DIAG_TRANSFORM
        )  # (T, n, 3, 3), (T, n, 3)
        phi6 = np.stack([phi[..., i, j] for i, j in PHI_COMPONENTS], axis=-1)  # (T, n, 6)
        return np.vstack([phi6.transpose(2, 1, 0), k.transpose(2, 1, 0)])

    phi, k = phi_analytical_strain(
        sd_degree, tmax, time_steps, s11=-1, s22=0.5, s33=0.5
    )  # (T, 6, n), (T, 3, n)
    return np.vstack([phi.transpose(1, 2, 0), k.transpose(1, 2, 0)])


def _phi_tensor(flat: np.ndarray) -> np.ndarray:
    """Expands flat state (9, n, T) into symmetric phi tensor (T, n, 3, 3)."""
    phi = np.zeros((flat.shape[-1], flat.shape[1], 3, 3))
    for component, (i, j) in enumerate(PHI_COMPONENTS):
        phi[:, :, i, j] = phi[:, :, j, i] = flat[component].T
    return phi


# ---------------------------------------------------------------------------
# Phi convergence (grid_convergence_all.svg)
# ---------------------------------------------------------------------------


def _collect_phi(case: str) -> tuple[list, list]:
    """Collects numerical and analytical phi tensors over TIME_STEPS_ARR."""
    grad_u, _ = CASES[case]
    tmax = _tmax(grad_u)

    num, anl = [], []
    for time_steps in tqdm(TIME_STEPS_ARR, desc=f"phi {case}"):
        num.append(_phi_tensor(_numerical(grad_u, SD_DEGREE, tmax, time_steps)))
        anl.append(_phi_tensor(_analytical(case, SD_DEGREE, tmax, time_steps)))
    return num, anl


def _phi_error(true: np.ndarray, predicted: np.ndarray) -> float:
    """
    Computes tensorially consistent relative phi error at the last time step.

    Args:
        true (np.ndarray): Analytical phi tensor, shape (T, n, 3, 3).
        predicted (np.ndarray): Numerical phi tensor, shape (T, n, 3, 3).

    Returns:
        float: Relative error at the last time step.
    """
    diff = true - predicted
    num = 4 * np.pi * np.einsum("tnij, tnij -> tn", diff, diff).mean(axis=1)
    den = 4 * np.pi * np.einsum("tnij, tnij -> tn", true, true).mean(axis=1)
    return float(num[-1] ** 0.5 / den[-1] ** 0.5)


def _plot_phi(ax: plt.Axes, case: str, num: list, anl: list) -> tuple[np.ndarray, list]:
    """Plots phi error vs. strain-time step, labelled with its fitted slope."""
    y = [_phi_error(a, n) for n, a in zip(num, anl)]
    x = ST_MAX / np.array(TIME_STEPS_ARR)
    slope, _ = np.polyfit(np.log(x), np.log(y), 1)
    ax.loglog(x, y, "o-", label=f"{case} (slope = {slope:.2f})")
    return x, y


def _add_reference_slope_line(
    ax: plt.Axes, x: np.ndarray, y: list, slope: float = 4, offset: float = 0.5
) -> None:
    """Draws a short solid slope-indicator segment with a rise/run annotation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    log_x = np.log10(x)
    log_range = log_x.max() - log_x.min()
    seg = log_range / 5
    log_xb = log_x.max() - 0.05 * log_range
    log_xa = log_xb - seg
    xa, xb = 10**log_xa, 10**log_xb

    order = np.argsort(log_x)
    ya = offset * 10 ** np.interp(log_xa, log_x[order], np.log10(y)[order])
    yb = ya * (xb / xa) ** slope

    xh0, xh1 = 10 ** (log_xa - seg / 4), 10 ** (log_xb + seg / 4)
    ax.loglog([xh0, xh1], [ya * (xh0 / xa) ** slope, ya * (xh1 / xa) ** slope], "k-", linewidth=0.9)
    ax.loglog([xa, xb], [ya, ya], "k-", linewidth=0.9)
    ax.loglog([xb, xb], [ya, yb], "k-", linewidth=0.9)
    ax.text(np.sqrt(xa * xb), 0.9 * ya, "1", ha="center", va="top")
    ax.text(1.05 * xb, np.sqrt(ya * yb), f"{slope}", ha="left", va="center")


def _format_log_xaxis(ax: plt.Axes) -> None:
    """Sets decimal-formatted log ticks on the x axis."""

    def decimal_log(value: float, _pos: int) -> str:
        if value == 0:
            return "0"
        if abs(value) < 1:
            return f"{value:.12f}".rstrip("0").rstrip(".")
        return f"{value:g}"

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(decimal_log))
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=12)
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


# ---------------------------------------------------------------------------
# M6 convergence (grid_convergence_M6_all.svg)
# ---------------------------------------------------------------------------


def _m6(flat: np.ndarray) -> np.ndarray:
    """Evaluates 6th-order M tensor, shape (3, 3, 3, 3, 3, 3, T), from flat state."""
    return evaluate_M_6th_order_es(flat[:6], flat[6:])


def _collect_m6(case: str, results_dir: Path) -> tuple[list, list, np.ndarray]:
    """
    Collects M6 tensors across SD_DEGREE_ARR, caching each to disk.

    Args:
        case (str): Case label, key of CASES.
        results_dir (Path): Directory holding the .npy caches.

    Returns:
        tuple[list, list, np.ndarray]: Numerical M6 per degree, analytical M6 per
            degree, and the analytical M6 at SD_DEGREE_TRUE.
    """
    grad_u, slug = CASES[case]
    tmax = _tmax(grad_u)

    num, anl = [], []
    for sd in tqdm(SD_DEGREE_ARR, desc=f"M6 {case}"):
        num.append(
            _cached(
                results_dir / f"M6_{slug}_sd{sd}_t{TIME_STEPS}.npy",
                lambda: _m6(_numerical(grad_u, sd, tmax, TIME_STEPS)),
            )
        )
        anl.append(
            _cached(
                results_dir / f"M6_{slug}_anl_sd{sd}_t{TIME_STEPS}.npy",
                lambda: _m6(_analytical(case, sd, tmax, TIME_STEPS)),
            )
        )

    truth = _cached(
        results_dir / f"M6_{slug}_anl_sd{SD_DEGREE_TRUE}_t{TIME_STEPS}.npy",
        lambda: _m6(_analytical(case, SD_DEGREE_TRUE, tmax, TIME_STEPS)),
    )
    return num, anl, truth


def _m6_error(num: list, reference) -> np.ndarray:
    """
    Computes relative L2 error of M6 at the last time step.

    Args:
        num (list): M6 tensors, one per spherical design degree.
        reference: Either a single M6 tensor, broadcast over all degrees, or one
            tensor per degree for a same-grid comparison.

    Returns:
        np.ndarray: Relative error per degree.
    """
    num = np.asarray(num)
    reference = np.broadcast_to(np.asarray(reference), num.shape)
    diff = num - reference
    idx = "nijpqrst, nijpqrst -> nt"
    numerator = np.einsum(idx, diff, diff) ** 0.5
    denominator = np.einsum(idx, reference, reference) ** 0.5
    return numerator[:, -1] / denominator[:, -1]


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------


def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    """Applies shared grid, tick, label, and legend styling."""
    ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def _save(path: Path) -> None:
    """Saves the current figure and logs where it went."""
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    logger.info("Plot saved at %s.", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(error_type: str = "quad_error") -> None:
    """
    Runs phi and M6 convergence studies for pure shear and AXE and saves plots.

    Args:
        error_type (str, optional): Interpretation of the M6 starred curves,
            "quad_error" (analytical at t vs. analytical at t=325, quadrature
            error) or "int_error" (numerical vs. analytical at the same t,
            time integration error). Defaults to "quad_error".

    Raises:
        ValueError: If error_type is not "quad_error" or "int_error".
    """
    if error_type not in ("int_error", "quad_error"):
        raise ValueError(
            f'Unknown error_type: {error_type!r}. Expected "int_error" or "quad_error".'
        )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results_dir = _results_dir()

    logger.info("Computing phi convergence.")
    _, ax = plt.subplots(figsize=(6, 6))
    curves = {case: _plot_phi(ax, case, *_collect_phi(case)) for case in CASES}
    _add_reference_slope_line(ax, *curves["AXE"], slope=4)
    _format_log_xaxis(ax)
    _style_axes(ax, r"$\Delta s$", r"$\mathcal{E}_{\phi}$")
    _save(results_dir / "grid_convergence_all.svg")

    logger.info("Computing M6 convergence.")
    m6 = {case: _collect_m6(case, results_dir) for case in CASES}

    _, ax = plt.subplots(figsize=(6, 6))
    for case, (num, _, truth) in m6.items():
        ax.semilogy(SD_DEGREE_ARR, _m6_error(num, truth), "o-", label=case)

    for case, (num, anl, truth) in m6.items():
        # int_error: numerical vs analytical on the same grid (time integration)
        # quad_error: analytical at t vs analytical at t = 325 (quadrature)
        series, reference = (num, anl) if error_type == "int_error" else (anl, truth)
        ax.semilogy(SD_DEGREE_ARR, _m6_error(series, reference), "s--", label=f"{case}*")

    _style_axes(ax, "t", r"$\mathcal{E}_{N}$")
    _save(results_dir / "grid_convergence_M6_all.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RDT-SD convergence plots.")
    parser.add_argument(
        "--error_type",
        type=str,
        default="quad_error",
        choices=["int_error", "quad_error"],
        help="Starred curves: quad_error = analytical at t vs analytical at t=325 (quadrature); "
        "int_error = numerical vs analytical at same t (time integration).",
    )
    args = parser.parse_args()
    main(args.error_type)
