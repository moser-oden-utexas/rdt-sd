"""Loads spherical design wavevector grids, caching them locally on first use."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UNSW_BASE_URL = "https://web.maths.unsw.edu.au/~rsw/Sphere/Points/SS/SS31-Mar-2016"
UNSW_SITE_URL = "https://web.maths.unsw.edu.au/~rsw/Sphere/Points/index.html"

# point counts that deviate from the general formula, per UNSW's own documentation
NUM_POINTS_EXCEPTIONS = {1: 2, 3: 6, 5: 12, 7: 32, 11: 70, 15: 120}


def _num_points(t: int) -> int:
    """
    Computes number of points in symmetric spherical design of degree t.

    Args:
        t (int): Spherical design degree, odd, 1 <= t <= 325.

    Returns:
        int: Number of points N in design.
    """
    if t in NUM_POINTS_EXCEPTIONS:
        return NUM_POINTS_EXCEPTIONS[t]
    return 2 * math.ceil((t**2 + t + 4) / 4)


def _grid_url(t: int) -> str:
    """
    Builds UNSW download URL for symmetric (SS) spherical design of degree t.

    Args:
        t (int): Spherical design degree, odd, 1 <= t <= 325.

    Returns:
        str: Download URL.
    """
    n = _num_points(t)
    return f"{UNSW_BASE_URL}/ss{t:03d}.{n:05d}"


def _cache_path(t: int) -> Path:
    """
    Resolves local cache path for grid of degree t.

    Args:
        t (int): Spherical design degree.

    Returns:
        Path: Path to cached grid file, rdt-sd/grids/sd{t}.
    """
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "grids" / f"sd{t}"


def _download_grid(t: int, dest: Path) -> None:
    """
    Downloads grid of degree t from UNSW and writes it to dest.

    Args:
        t (int): Spherical design degree.
        dest (Path): Local path to write downloaded grid to.

    Raises:
        RuntimeError: If download fails for any reason.
    """
    url = _grid_url(t)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to download spherical design grid for t={t} from {url}. "
            f"Underlying error: {e}. Spherical designs are hosted at {UNSW_SITE_URL}; "
            f"if the site is unavailable, source the grid manually and place it at {dest}."
        ) from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(response.text)


def init_wavenumbers_spherical_designs(t: int = 109) -> tuple[np.ndarray, int]:
    """
    Initializes wavenumbers using spherical designs.

    Checks rdt-sd/grids/ for a locally cached grid first; on a cache miss,
    downloads the grid from UNSW's spherical designs site and caches it for
    subsequent calls.

    Args:
        t (int, optional): Degree to which spherical harmonics are
            integrated exactly using spherical designs, odd, 1 <= t <= 325.
            Defaults to 109.

    Returns:
        tuple[np.ndarray, int]: karr with shape (num_nodes, 3), the
            spherical-design-initialized wavenumber space, and num_nodes.

    Raises:
        RuntimeError: If grid is not cached locally and downloading it fails.
    """
    assert t % 2 == 1 and 1 <= t <= 325, "t must be an odd integer with 1 <= t <= 325."

    cache_path = _cache_path(t)
    if not cache_path.exists():
        _download_grid(t, cache_path)

    data = pd.read_csv(cache_path, sep=r"\s+", header=None, names=["x", "y", "z"])

    num_nodes = len(data)
    karr = np.zeros([num_nodes, 3])
    karr[:, 0] = np.array(data["x"])
    karr[:, 1] = np.array(data["y"])
    karr[:, 2] = np.array(data["z"])

    return karr, num_nodes
