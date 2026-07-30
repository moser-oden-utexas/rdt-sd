"""
Pushes axisymmetric turbulence toward the one-component limit with frame rotation.

Ten cases share one axisymmetric velocity gradient and differ only in the
rotation rate about e2. As |Omega| grows, the anisotropy trajectory is driven
toward the 1C corner of the barycentric triangle.

Runs the ensemble, evaluates R_ij / D_ij and the early-stopping index, and
caches them so repeated runs only redo the plotting. Pass --regenerate to
discard the cache, which is needed whenever the solver or the underlying
spherical design grid changes.

Outputs:
  results/push_1c/push_to_1c_bij.svg
  results/push_1c/push_to_1c_yij.svg
"""

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.barycentric_plots import barycentric_map_outline, barycentric_map_point
from src.earlystopping import batched_stopping_index
from src.rdt_solver import simulate_parallel
from src.tensor_utils import get_D_ij, get_R_ij, strain_rate

logger = logging.getLogger(__name__)

# axisymmetric case, rotated about e2 at ten increasing rates
GRAD_U = 2*np.array([[1, 0, 0], [0, -1/2, 0], [0, 0, -1/2]])
OMEGA_2 =  np.array([0, 0.25, 0.75, 1.5, 3, 5, 6, 7, 8, 10])

ST_MAX = 10
NUM_TIME_STEPS = 100
SD_DEGREE = 109 
SOLVER = "dopri5"

# early stopping, matching the convention in scripts/earlystopping_thresholds.py
ES_DEGREE = (SD_DEGREE // 4) * 2
ES_THRESHOLD = 3e-4 #1.6e-4

# realizability bounds on the eigenvalues of an anisotropy tensor
EIGENVALUE_BOUNDS = (-1 / 3, 2 / 3)

COLORMAP = "turbo"
NUM_COLORBAR_TICKS = 6

# style for the segment of each trajectory after its early-stopping index
DISCARDED_COLOR = "0.75"
DISCARDED_ALPHA = 0.5
DISCARDED_LINEWIDTH = 0.8

# fixed view limits for the triangle axes, identical across plots regardless of
# the corner label ("C" vs "D") so the two figures scale identically
TRIANGLE_XLIM = (-0.25, 1.15)
TRIANGLE_YLIM = (-0.08, 1.0)

# physical layout, in inches: the triangle axes are this size in every plot, so
# a colorbar beside one never shrinks it relative to a plot without one. Height
# is derived to match the data aspect ratio exactly, so requesting the box at
# this size already satisfies set_aspect('equal') with no leftover shrink-gap
# (which would otherwise show up as blank margin and as a colorbar taller than
# the triangle it sits beside).
TRIANGLE_SIZE_IN = 5.0
_TRIANGLE_DATA_ASPECT = (TRIANGLE_YLIM[1] - TRIANGLE_YLIM[0]) / (TRIANGLE_XLIM[1] - TRIANGLE_XLIM[0])
TRIANGLE_HEIGHT_IN = TRIANGLE_SIZE_IN * _TRIANGLE_DATA_ASPECT

MARGIN_IN = 0.15
COLORBAR_WIDTH_IN = 0.25
COLORBAR_PAD_IN = 0.2
COLORBAR_LABEL_SPACE_IN = 0.7
COLORBAR_HEIGHT_FRACTION = 0.8  # of TRIANGLE_HEIGHT_IN, centered on the triangle's span
SAVE_PAD_IN = 0.03


def _results_dir() -> Path:
    """Resolves and creates rdt-sd/results/push_1c."""
    results_dir = Path(__file__).resolve().parents[1] / "results" / "push_1c"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _generate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the rotation sweep and reduces it to the quantities the plots need.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: R_ij and D_ij, each shape
            (3, 3, num_cases, num_time_steps), and the early-stopping index per
            case, shape (num_cases,).
    """
    num_cases = len(OMEGA_2)
    grad_u = np.repeat(GRAD_U[None, ...], num_cases, axis=0)
    omega = np.zeros((num_cases, 3))
    omega[:, 1] = OMEGA_2
    tmax = ST_MAX / strain_rate(grad_u)

    logger.info("Simulating %d rotation rates.", num_cases)
    sol, _ = simulate_parallel(grad_u, tmax, NUM_TIME_STEPS, omega, SD_DEGREE, solver=SOLVER)
    sol = np.asarray(sol)  # (num_cases, T, 9, num_nodes)

    logger.info("Evaluating early stopping at degree %d.", ES_DEGREE)
    es_array = batched_stopping_index(sol, degree=ES_DEGREE, thr=ES_THRESHOLD)

    return get_R_ij(sol), get_D_ij(sol), es_array


def _load_or_generate(results_dir: Path, regenerate: bool) -> tuple[np.ndarray, ...]:
    """
    Loads the cached sweep, or runs it and caches the result.

    Args:
        results_dir (Path): Directory holding the cache.
        regenerate (bool): Ignores any existing cache when True.

    Returns:
        tuple[np.ndarray, ...]: R_ij, D_ij, and the early-stopping indices.
    """
    cache_path = results_dir / "push_to_1c_data.npz"

    if cache_path.exists() and not regenerate:
        logger.info("Loading cached sweep from %s.", cache_path)
        cached = np.load(cache_path)
        return cached["r_ij"], cached["d_ij"], cached["es_array"]

    r_ij, d_ij, es_array = _generate()
    np.savez(cache_path, r_ij=r_ij, d_ij=d_ij, es_array=es_array)
    logger.info("Cached sweep at %s.", cache_path)
    return r_ij, d_ij, es_array


def _anisotropy(tensor_ij: np.ndarray, reference_ij: np.ndarray | None = None) -> np.ndarray:
    """
    Computes the anisotropy of a second-order tensor.

    Normalizes by the trace of reference_ij, so D_ij can be normalized by the
    trace of R_ij rather than its own.

    Args:
        tensor_ij (np.ndarray): Second-order tensor, shape (3, 3, batch, time).
        reference_ij (np.ndarray | None, optional): Tensor supplying the
            normalizing trace. Defaults to None, which uses tensor_ij itself.

    Returns:
        np.ndarray: Anisotropy tensor, shape (3, 3, batch, time).
    """
    if reference_ij is None:
        reference_ij = tensor_ij

    trace = np.einsum("iibt -> bt", reference_ij)[None, None, :, :]
    return tensor_ij / trace - np.eye(3)[..., None, None] / 3


def _barycentric_track(
    aij: np.ndarray, case_index: int, start: int, stop: int
) -> tuple[list, list]:
    """
    Maps one case's realizable anisotropy states onto the barycentric triangle.

    States whose eigenvalues fall outside EIGENVALUE_BOUNDS are unphysical and
    are skipped.

    Args:
        aij (np.ndarray): Anisotropy tensor, shape (3, 3, batch, time).
        case_index (int): Index of the case along the batch axis.
        start (int): First time step to include.
        stop (int): Time steps at or beyond this index are excluded.

    Returns:
        tuple[list, list]: Barycentric x and y coordinates of realizable states.
    """
    lower, upper = EIGENVALUE_BOUNDS
    xs, ys = [], []

    for time_step in range(start, stop):
        state = aij[..., case_index, time_step]
        evals = np.real(np.linalg.eigvals(state))

        if not np.all((lower < evals) & (evals < upper)):
            continue

        x, y = barycentric_map_point(state, plot=False)
        xs.append(x)
        ys.append(y)

    return xs, ys


def _plot(
    aij: np.ndarray, es_array: np.ndarray, label: str, out_path: Path, show_colorbar: bool
) -> None:
    """
    Draws one anisotropy trajectory per rotation rate, coloured by |Omega|.

    Each trajectory is split at its early-stopping index: the trusted segment
    before it is coloured by rotation rate, and the discarded segment at or
    beyond it is drawn in flat grey, sharing its first point with the trusted
    segment's last so the two connect with no gap.

    The triangle axes are a fixed physical size (TRIANGLE_SIZE_IN x
    TRIANGLE_HEIGHT_IN) regardless of show_colorbar, so this plot and its
    counterpart with the opposite show_colorbar value render the triangle at
    the same size — a colorbar only ever adds canvas beside the triangle, it
    never shrinks it. The colorbar is drawn at COLORBAR_HEIGHT_FRACTION of the
    triangle's height, centered on its span. The saved figure is cropped
    tightly to its content via bbox_inches, so this keeps the two plots
    visually matched when laid out side by side.

    Args:
        aij (np.ndarray): Anisotropy tensor, shape (3, 3, batch, time).
        es_array (np.ndarray): Early-stopping index per case, shape (batch,).
        label (str): Corner label of the barycentric triangle, "C" or "D".
        out_path (Path): Output svg path.
        show_colorbar (bool): Draws the |Omega| colorbar when True.
    """
    num_time_steps = aij.shape[-1]
    omega_mag = np.abs(OMEGA_2)
    cmap = plt.get_cmap(COLORMAP)
    # colour by rotation magnitude, so the colorbar reads in physical units
    norm = mpl.colors.Normalize(vmin=0.0, vmax=omega_mag.max())

    extra_width = (
        COLORBAR_PAD_IN + COLORBAR_WIDTH_IN + COLORBAR_LABEL_SPACE_IN if show_colorbar else 0.0
    )
    fig_width = TRIANGLE_SIZE_IN + 2 * MARGIN_IN + extra_width
    fig_height = TRIANGLE_HEIGHT_IN + 2 * MARGIN_IN

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(
        [
            MARGIN_IN / fig_width,
            MARGIN_IN / fig_height,
            TRIANGLE_SIZE_IN / fig_width,
            TRIANGLE_HEIGHT_IN / fig_height,
        ]
    )
    plt.sca(ax)
    barycentric_map_outline(label=label)

    for case_index in range(aij.shape[-2]):
        stop = es_array[case_index]

        xs_discarded, ys_discarded = _barycentric_track(
            aij, case_index, max(stop - 1, 0), num_time_steps
        )
        if xs_discarded:
            ax.plot(
                xs_discarded,
                ys_discarded,
                "-o",
                color=DISCARDED_COLOR,
                linewidth=DISCARDED_LINEWIDTH,
                markersize=3.0,
                markerfacecolor=DISCARDED_COLOR,
                markeredgecolor=DISCARDED_COLOR,
                alpha=DISCARDED_ALPHA,
                zorder=1,
            )

    for case_index in range(aij.shape[-2]):
        stop = es_array[case_index]

        xs, ys = _barycentric_track(aij, case_index, 0, stop)
        if not xs:
            continue
        color = cmap(norm(omega_mag[case_index]))
        ax.plot(
            xs,
            ys,
            "-o",
            color=color,
            linewidth=1.0,
            markersize=3.0,
            markerfacecolor=color,
            markeredgecolor=color,
            alpha=0.9,
            zorder=2,
        )

    if show_colorbar:
        scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        colorbar_height_in = TRIANGLE_HEIGHT_IN * COLORBAR_HEIGHT_FRACTION
        colorbar_bottom_in = MARGIN_IN + (TRIANGLE_HEIGHT_IN - colorbar_height_in) / 2
        cax = fig.add_axes(
            [
                (MARGIN_IN + TRIANGLE_SIZE_IN + COLORBAR_PAD_IN) / fig_width,
                colorbar_bottom_in / fig_height,
                COLORBAR_WIDTH_IN / fig_width,
                colorbar_height_in / fig_height,
            ]
        )
        colorbar = fig.colorbar(scalar_map, cax=cax)
        colorbar.set_label(r"$|\Omega|$", fontsize=20)
        colorbar.set_ticks(np.linspace(0.0, omega_mag.max(), NUM_COLORBAR_TICKS))

    ax.set_xlim(TRIANGLE_XLIM)
    ax.set_ylim(TRIANGLE_YLIM)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=SAVE_PAD_IN)
    plt.close(fig)

    logger.info("Plot saved at %s.", out_path)


def main(regenerate: bool = False) -> None:
    """
    Plots R_ij and D_ij anisotropies for the rotation sweep on barycentric maps.

    Args:
        regenerate (bool, optional): Discards the cached sweep and reruns the
            solver. Defaults to False.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results_dir = _results_dir()

    r_ij, d_ij, es_array = _load_or_generate(results_dir, regenerate)
    logger.info("Stopping indices: %s.", es_array.tolist())

    _plot(
        _anisotropy(r_ij), es_array, "C", results_dir / "push_to_1c_bij.svg", show_colorbar=False
    )
    _plot(
        _anisotropy(d_ij, reference_ij=r_ij),
        es_array,
        "D",
        results_dir / "push_to_1c_yij.svg",
        show_colorbar=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the push-to-1C rotation sweep.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Discard the cached sweep and rerun the solver.",
    )
    args = parser.parse_args()
    main(args.regenerate)
