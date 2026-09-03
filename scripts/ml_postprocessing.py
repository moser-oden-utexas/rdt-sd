"""
Builds ML-ready arrays from postprocessed structure tensors.

Reads the structure tensors and es_array written by scripts/postprocessing.py,
normalizes every tensor by q2, truncates each case at its early-stopping index,
flattens the case and time axes into a single sample axis, attaches the per-case
mean velocity gradients and Coriolis terms reconstructed from the launcher config
that generated the run, and drops unrealizable samples.

Outputs, under <--output_dir>:
  one npy per array, all sharing a trailing sample axis: q2, r_ij, d_ij, q_ijk,
    qs_ijk, m_ijpq, ms_ijpq, l_ijpq, j_ijrpq, gradU, and corU when the run used
    Coriolis.
  anisotropy_plot_Rij.svg and anisotropy_plot_Dij.svg, showing the spread of the
    dataset's anisotropy states on the barycentric (Lumley) triangle.
"""

import argparse
import logging
import pickle
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.barycentric_plots import (
    anisotropy,
    barycentric_map_outline,
    barycentric_map_point,
)
from src.sampler import resolve_case_parameters

logger = logging.getLogger(__name__)

# slack on the realizability bounds of an anisotropy eigenvalue
REALIZABILITY_TOLERANCE = 1e-10

# realizability bounds on the eigenvalues of an anisotropy tensor
EIGENVALUE_BOUNDS = (-1 / 3, 2 / 3)

# every BARY_PLOTTING_FREQUENCY-th sample is drawn on the barycentric maps
BARY_PLOTTING_FREQUENCY = 50


def load_normalized_tensors(structure_tensors: Path) -> dict[str, np.ndarray]:
    """
    Loads saved structure tensors and normalizes them by q2.

    Keys are lowercased, so the saved R_ij becomes r_ij. q2 broadcasts over the
    trailing (case, time) axes of every tensor rank and is itself left unnormalized.

    Args:
        structure_tensors (Path): Pickled dict from compute_structure_tensors, each
            value shaped (..., num_cases, num_time_steps).

    Returns:
        dict[str, np.ndarray]: Normalized tensors, each shape
            (..., num_cases, num_time_steps).

    Raises:
        KeyError: If the dict does not contain q2.
        ValueError: If q2 is not two-dimensional, or a tensor's trailing axes do
            not match q2.
    """
    with open(structure_tensors, "rb") as f:
        tensors = pickle.load(f)

    if "q2" not in tensors:
        raise KeyError(f"{structure_tensors} does not contain 'q2'.")

    q2 = np.asarray(tensors["q2"])
    if q2.ndim != 2:
        raise ValueError(
            f"q2 must have shape (num_cases, num_time_steps), got {q2.shape}."
        )

    normalized = {}
    for key, tensor in tensors.items():
        tensor = np.asarray(tensor)
        if tensor.shape[-2:] != q2.shape:
            raise ValueError(
                f"{key} has trailing axes {tensor.shape[-2:]}, expected {q2.shape} "
                "to match q2."
            )
        normalized[key.lower()] = q2 if key == "q2" else tensor / q2

    return normalized


def case_parameters_from_config(config: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Reconstructs per-case mean velocity gradients and Coriolis terms.

    Repeats the case resolution done by scripts/launcher.py, which does not save
    the resolved values. Cases are re-sampled from the Sobol sequence, or reloaded
    from `grad_u_location`/`coriolis_location` when the config carries them. When
    the config carries `case_offset`/`total_num_samples` (written for a run that is
    one disjoint slice of a larger shared pool, e.g. scripts/build_ml_datasets.py),
    the full pool is resolved and then sliced down to this run's `num_samples`,
    starting at `case_offset`. Both default to single-run behavior (offset 0, total
    equal to num_samples) when absent.

    Args:
        config (Path): Launcher toml config used to generate the run.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Mean velocity gradients, shape
            (num_cases, 3, 3), and Coriolis terms, shape (num_cases, 3), or None
            when the run did not use Coriolis.

    Raises:
        ValueError: If the config does not describe an ensemble run, or if the
            resolved pool does not cover this run's slice.
    """
    with open(config, "rb") as f:
        launcher_config = tomllib.load(f)

    if not launcher_config["run_type"]["ensemble"]:
        raise ValueError("Config must describe an ensemble run.")

    ensemble = launcher_config["ensemble"]
    use_coriolis = ensemble["use_coriolis"]
    num_samples = ensemble["num_samples"]
    total_num_samples = ensemble.get("total_num_samples", num_samples)
    case_offset = ensemble.get("case_offset", 0)

    mean_velocity_gradients, coriolis_terms = resolve_case_parameters(
        total_num_samples,
        use_coriolis,
        seed=ensemble["seed"],
        grad_u_location=ensemble.get("grad_u_location"),
        coriolis_location=ensemble.get("coriolis_location"),
    )

    if mean_velocity_gradients.shape[0] < case_offset + num_samples:
        raise ValueError(
            f"Resolved case pool holds {mean_velocity_gradients.shape[0]} cases, but "
            f"this run covers cases [{case_offset}, {case_offset + num_samples})."
        )

    case_slice = slice(case_offset, case_offset + num_samples)
    mean_velocity_gradients = mean_velocity_gradients[case_slice]
    coriolis_terms = coriolis_terms[case_slice] if coriolis_terms is not None else None

    return mean_velocity_gradients, coriolis_terms


def validate_es_array(
    es_array: np.ndarray, num_cases: int, num_time_steps: int
) -> np.ndarray:
    """
    Checks that early-stopping indices match the structure tensors.

    Args:
        es_array (np.ndarray): Early-stopping index per case, shape (num_cases,).
        num_cases (int): Number of cases covered by the structure tensors.
        num_time_steps (int): Number of time steps per case.

    Returns:
        np.ndarray: Validated early-stopping indices, shape (num_cases,), dtype int.

    Raises:
        ValueError: If es_array is not one-dimensional, does not cover every case,
            or holds an index outside [0, num_time_steps].
    """
    if es_array.ndim != 1:
        raise ValueError(f"es_array must be one-dimensional, got {es_array.shape}.")
    if es_array.shape[0] != num_cases:
        raise ValueError(
            f"es_array covers {es_array.shape[0]} cases, but the structure tensors "
            f"cover {num_cases}."
        )

    es_array = es_array.astype(int)
    if np.any(es_array < 0) or np.any(es_array > num_time_steps):
        raise ValueError(
            f"es_array holds indices outside [0, {num_time_steps}]: "
            f"[{es_array.min()}, {es_array.max()}]."
        )

    return es_array


def trim_and_flatten(
    tensors: dict[str, np.ndarray], es_array: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Truncates each case at its early-stopping index and merges case and time.

    Args:
        tensors (dict[str, np.ndarray]): Tensors, each shape
            (..., num_cases, num_time_steps).
        es_array (np.ndarray): Early-stopping index per case, shape (num_cases,).

    Returns:
        dict[str, np.ndarray]: Tensors, each shape (..., num_samples).
    """
    return {
        name: np.concatenate(
            [tensor[..., case, :stop] for case, stop in enumerate(es_array)], axis=-1
        )
        for name, tensor in tensors.items()
    }


def repeat_case_values(case_array: np.ndarray, es_array: np.ndarray) -> np.ndarray:
    """
    Repeats each per-case value over the time steps retained for that case.

    Args:
        case_array (np.ndarray): Per-case values, shape (num_cases, ...), e.g.
            (n, 3, 3) mean velocity gradients or (n, 3) Coriolis terms.
        es_array (np.ndarray): Early-stopping index per case, shape (num_cases,).

    Returns:
        np.ndarray: Repeated values, shape (..., num_samples).
    """
    return np.concatenate(
        [
            np.repeat(case_array[case, ..., None], repeats=stop, axis=-1)
            for case, stop in enumerate(es_array)
        ],
        axis=-1,
    )


def filter_realizable(flattened: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Drops samples whose R_ij or D_ij anisotropy is not realizable.

    Both anisotropies take their normalizing trace from r_ij, which is q2 and is
    therefore already one. Samples with non-finite entries or with an eigenvalue
    outside EIGENVALUE_BOUNDS are unphysical and are removed from every array.

    Args:
        flattened (dict[str, np.ndarray]): Tensors, each shape (..., num_samples),
            including r_ij and d_ij shaped (3, 3, num_samples).

    Returns:
        dict[str, np.ndarray]: Tensors, each shape (..., num_realizable_samples).

    Raises:
        KeyError: If r_ij or d_ij is missing.
    """
    if "r_ij" not in flattened or "d_ij" not in flattened:
        raise KeyError("Flattened tensors must contain 'r_ij' and 'd_ij'.")

    r_ij = flattened["r_ij"]
    num_samples = r_ij.shape[-1]

    b_ij = anisotropy(r_ij)
    y_ij = anisotropy(flattened["d_ij"], reference_ij=r_ij)

    finite = np.all(np.isfinite(b_ij), axis=(0, 1)) & np.all(
        np.isfinite(y_ij), axis=(0, 1)
    )
    finite_indices = np.flatnonzero(finite)

    # eigvals needs finite entries, so only the finite samples are evaluated
    lower, upper = EIGENVALUE_BOUNDS
    keep = np.zeros(num_samples, dtype=bool)
    if finite_indices.size:
        keep_finite = np.ones(finite_indices.size, dtype=bool)
        for anisotropy_ij in (b_ij, y_ij):
            evals = np.real(
                np.linalg.eigvals(np.moveaxis(anisotropy_ij[..., finite_indices], -1, 0))
            )
            keep_finite &= np.all(
                (evals >= lower - REALIZABILITY_TOLERANCE)
                & (evals <= upper + REALIZABILITY_TOLERANCE),
                axis=-1,
            )
        keep[finite_indices] = keep_finite

    retained = int(np.count_nonzero(keep))
    logger.info(
        "Realizability filtering retained %d and removed %d samples.",
        retained,
        num_samples - retained,
    )

    return {name: tensor[..., keep] for name, tensor in flattened.items()}


def plot_barycentric(anisotropy_ij: np.ndarray, label: str, output_path: Path) -> None:
    """
    Draws the dataset's anisotropy states on the barycentric map and saves it.

    Every BARY_PLOTTING_FREQUENCY-th sample is drawn, which keeps the figure
    readable for datasets running to tens of thousands of samples. Samples are
    ordered case-major, so the stride walks the whole ensemble. No realizability
    check is needed, as filter_realizable has already dropped every unphysical
    sample.

    Args:
        anisotropy_ij (np.ndarray): Anisotropy tensor, shape (3, 3, num_samples).
        label (str): Corner label of the barycentric triangle, "C" or "D".
        output_path (Path): Output svg path.
    """
    subsampled = anisotropy_ij[..., ::BARY_PLOTTING_FREQUENCY]
    points = [
        barycentric_map_point(subsampled[..., sample], plot=False)
        for sample in range(subsampled.shape[-1])
    ]
    xs, ys = zip(*points)

    plt.figure(figsize=(5, 5))
    barycentric_map_outline(label=label)
    plt.plot(xs, ys, ".", color="0.1", alpha=0.5, linewidth=0.8)

    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()

    logger.info("Saved %s with %d points.", output_path, len(points))


def main(
    structure_tensors: Path, es_array: Path, config: Path, output_dir: Path
) -> None:
    """
    Assembles and saves the ML arrays and anisotropy plots for one ensemble.

    Args:
        structure_tensors (Path): Pickled structure tensors from scripts/postprocessing.py.
        es_array (Path): Npy early-stopping indices from scripts/postprocessing.py.
        config (Path): Launcher toml config used to generate the run.
        output_dir (Path): Destination directory for the arrays and plots.

    Raises:
        ValueError: If the config covers a different number of cases than the
            structure tensors, or if the assembled arrays disagree on sample count.
    """
    normalized = load_normalized_tensors(structure_tensors)
    num_cases, num_time_steps = normalized["q2"].shape

    stops = validate_es_array(np.load(es_array), num_cases, num_time_steps)

    mean_velocity_gradients, coriolis_terms = case_parameters_from_config(config)
    if mean_velocity_gradients.shape[0] != num_cases:
        raise ValueError(
            f"Config resolves {mean_velocity_gradients.shape[0]} cases, but the "
            f"structure tensors cover {num_cases}."
        )

    flattened = trim_and_flatten(normalized, stops)
    flattened["gradU"] = repeat_case_values(mean_velocity_gradients, stops)
    if coriolis_terms is not None:
        flattened["corU"] = repeat_case_values(coriolis_terms, stops)

    flattened = filter_realizable(flattened)

    num_samples = flattened["r_ij"].shape[-1]
    for name, tensor in flattened.items():
        if tensor.shape[-1] != num_samples:
            raise ValueError(
                f"{name} holds {tensor.shape[-1]} samples, expected {num_samples}."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in flattened.items():
        np.save(output_dir / f"{name}.npy", tensor)
        logger.info("Saved %s with shape %s.", name, tensor.shape)

    r_ij = flattened["r_ij"]
    plot_barycentric(anisotropy(r_ij), "C", output_dir / "anisotropy_plot_Rij.svg")
    plot_barycentric(
        anisotropy(flattened["d_ij"], reference_ij=r_ij),
        "D",
        output_dir / "anisotropy_plot_Dij.svg",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ML-ready arrays from postprocessed structure tensors."
    )
    parser.add_argument(
        "--structure_tensors",
        type=Path,
        required=True,
        help="Pickled structure tensors written by scripts/postprocessing.py.",
    )
    parser.add_argument(
        "--es_array",
        type=Path,
        required=True,
        help="Npy early-stopping indices written by scripts/postprocessing.py.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Launcher toml config used to generate the run.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination directory for the saved arrays.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(args.structure_tensors, args.es_array, args.config, args.output_dir)
