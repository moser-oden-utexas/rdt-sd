"""
Barycentric anisotropy plots for the coriolis and no-coriolis rdt ensembles.

Reads the cached structure tensors of the 200-case ensembles in results/runs,
maps the R_ij and D_ij anisotropies onto the barycentric (Lumley) triangle, and
truncates each case at its early-stopping index.

Outputs:
  results/barycentric_plots/coriolis_rdt_jcp_anisotropy_plot_Rij.svg
  results/barycentric_plots/coriolis_rdt_jcp_anisotropy_plot_Dij.svg
  results/barycentric_plots/no_coriolis_rdt_jcp_anisotropy_plot_Rij.svg
  results/barycentric_plots/no_coriolis_rdt_jcp_anisotropy_plot_Dij.svg
"""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.barycentric_plots import (
    anisotropy,
    barycentric_map_outline,
    barycentric_map_point,
)

logger = logging.getLogger(__name__)

CASES = ("coriolis_rdt_jcp", "no_coriolis_rdt_jcp")

# every SUBSAMPLE-th realizable state along a trajectory is drawn
SUBSAMPLE = 6

# realizability bounds on the eigenvalues of an anisotropy tensor
EIGENVALUE_BOUNDS = (-1 / 3, 2 / 3)


def _runs_dir() -> Path:
    """Resolves rdt-sd/results/runs, holding the cached ensembles."""
    return Path(__file__).resolve().parents[1] / "results" / "runs"


def _results_dir() -> Path:
    """Resolves and creates rdt-sd/results/barycentric_plots."""
    results_dir = Path(__file__).resolve().parents[1] / "results" / "barycentric_plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _load(case: str) -> tuple[dict, np.ndarray]:
    """
    Loads cached structure tensors and early-stopping indices for one ensemble.

    Args:
        case (str): Ensemble name, one of CASES.

    Returns:
        tuple[dict, np.ndarray]: Structure tensors, each shape (..., batch, time),
            and early-stopping index per case, shape (batch,).

    Raises:
        FileNotFoundError: If either cached input is missing.
    """
    runs_dir = _runs_dir()
    tensors_path = runs_dir / f"structure_tensors_{case}.pkl"
    es_path = runs_dir / f"{case}_es_array.npy"

    for path in (tensors_path, es_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing cached ensemble input: {path}.")

    with open(tensors_path, "rb") as f:
        tensors = pickle.load(f)

    return tensors, np.load(es_path)


def _barycentric_track(aij: np.ndarray, case_index: int, stop: int) -> tuple[list, list]:
    """
    Maps one case's realizable anisotropy states onto the barycentric triangle.

    States whose eigenvalues fall outside EIGENVALUE_BOUNDS are unphysical and
    are skipped.

    Args:
        aij (np.ndarray): Anisotropy tensor, shape (3, 3, batch, time).
        case_index (int): Index of the case along the batch axis.
        stop (int): Early-stopping index; states at or beyond it are ignored.

    Returns:
        tuple[list, list]: Barycentric x and y coordinates of realizable states.
    """
    lower, upper = EIGENVALUE_BOUNDS
    xs, ys = [], []

    for time_step in range(stop):
        state = aij[..., case_index, time_step]
        evals = np.real(np.linalg.eigvals(state))

        if not np.all((lower < evals) & (evals < upper)):
            continue

        x, y = barycentric_map_point(state, plot=False)
        xs.append(x)
        ys.append(y)

    return xs, ys


def _plot_case(aij: np.ndarray, es_array: np.ndarray, label: str, out_path: Path) -> None:
    """
    Draws every case's anisotropy trajectory on the barycentric map and saves it.

    Args:
        aij (np.ndarray): Anisotropy tensor, shape (3, 3, batch, time).
        es_array (np.ndarray): Early-stopping index per case, shape (batch,).
        label (str): Corner label of the barycentric triangle, "C" or "D".
        out_path (Path): Output svg path.
    """
    plt.figure(figsize=(5, 5))
    barycentric_map_outline(label=label)

    for case_index in range(aij.shape[-2]):
        xs, ys = _barycentric_track(aij, case_index, es_array[case_index])
        if xs:
            plt.plot(
                xs[::SUBSAMPLE], ys[::SUBSAMPLE], ".", color="0.1", alpha=0.5, linewidth=0.8
            )

    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()

    logger.info("Plot saved at %s.", out_path)


def main() -> None:
    """Plots R_ij and D_ij anisotropies for each ensemble on barycentric maps."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results_dir = _results_dir()

    for case in CASES:
        logger.info("Plotting %s.", case)
        tensors, es_array = _load(case)

        r_ij = tensors["R_ij"]
        b_ij = anisotropy(r_ij)
        y_ij = anisotropy(tensors["D_ij"], reference_ij=r_ij)

        _plot_case(b_ij, es_array, "C", results_dir / f"{case}_anisotropy_plot_Rij.svg")
        _plot_case(y_ij, es_array, "D", results_dir / f"{case}_anisotropy_plot_Dij.svg")


if __name__ == "__main__":
    main()
