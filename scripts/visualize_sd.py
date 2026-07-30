"""
Visualizations of spherical t-design wavevector grids.

Renders each requested degree as a 3D scatter of grid points on an opaque
unit sphere.

Outputs:
  results/spherical_designs/k_spherical_designs_t{t}.svg
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.spherical_designs import init_wavenumbers_spherical_designs

logger = logging.getLogger(__name__)

DEFAULT_DEGREES = [5, 19, 45]
SPHERE_RADIUS = 0.987
AXIS_LIMIT = 1.05  # grid points lie on the unit sphere; keep a small margin so none clip
ELEV, AZIM = 45, 0
BACK_ALPHA = 0.25
ZOOM = 1.5  # fills more of the frame before bbox_inches="tight" crops to content


def _results_dir() -> Path:
    """Resolves and creates rdt-sd/results/spherical_designs."""
    results_dir = Path(__file__).resolve().parents[1] / "results" / "spherical_designs"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _plot_sphere(ax: plt.Axes) -> None:
    """Draws opaque unit-sphere surface, slightly under the grid points' radius."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u) * SPHERE_RADIUS, np.cos(v))
    ax.plot_surface(x, y, z, alpha=1, color="lightblue", zorder=1)


def _point_alphas(karr: np.ndarray) -> np.ndarray:
    """
    Assigns per-point opacity: full on the camera-facing hemisphere, reduced
    on the far side.

    The sphere surface and scatter are drawn with `computed_zorder=False`
    (see `_plot_degree`), so the scatter always draws on top of the surface
    by draw order — mplot3d's default automatic depth sorting is unreliable
    between an opaque surface and a point collection. With true occlusion
    unavailable, far-side points are faded instead, using the same view
    direction passed to `ax.view_init`, so they still read as "behind" the
    sphere rather than sitting on top of it at full strength.

    Args:
        karr (np.ndarray): Unit wavevectors, shape (num_nodes, 3).

    Returns:
        np.ndarray: Per-point alpha, shape (num_nodes,).
    """
    view_dir = np.array(
        [
            np.cos(np.radians(ELEV)) * np.cos(np.radians(AZIM)),
            np.cos(np.radians(ELEV)) * np.sin(np.radians(AZIM)),
            np.sin(np.radians(ELEV)),
        ]
    )
    front = karr @ view_dir > 0
    return np.where(front, 1.0, BACK_ALPHA)


def _plot_degree(t: int, results_dir: Path) -> None:
    """
    Plots and saves the spherical t-design grid of degree t.

    Args:
        t (int): Spherical design degree, odd, 1 <= t <= 325.
        results_dir (Path): Directory to save the .svg into.
    """
    karr, num_nodes = init_wavenumbers_spherical_designs(t)
    logger.info("t = %d: %d nodes.", t, num_nodes)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    _plot_sphere(ax)
    ax.scatter(
        karr[:, 0],
        karr[:, 1],
        karr[:, 2],
        c="k",
        s=10,
        depthshade=True,
        zorder=2,
        alpha=_point_alphas(karr),
    )

    ax.set_box_aspect([1, 1, 1], zoom=ZOOM)
    ax.set_xlim([-AXIS_LIMIT, AXIS_LIMIT])
    ax.set_ylim([-AXIS_LIMIT, AXIS_LIMIT])
    ax.set_zlim([-AXIS_LIMIT, AXIS_LIMIT])
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    path = results_dir / f"k_spherical_designs_t{t}.svg"
    plt.savefig(path, dpi=300, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved at %s.", path)


def main(degrees: list[int]) -> None:
    """
    Renders and saves a spherical t-design grid plot for each degree.

    Args:
        degrees (list[int]): Spherical design degrees to render, each odd
            with 1 <= t <= 325.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results_dir = _results_dir()
    for t in degrees:
        _plot_degree(t, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize spherical t-design grids.")
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=DEFAULT_DEGREES,
        help="Spherical design degrees to render, each odd with 1 <= t <= 325.",
    )
    args = parser.parse_args()
    main(args.degrees)
