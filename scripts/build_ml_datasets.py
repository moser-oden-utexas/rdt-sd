"""
Builds train + n test ML datasets spanning different St ranges.

Draws one training dataset integrated to `--st_train` and one testing dataset
per `--st_test` value, each integrated further. All `train_num_cases + n *
test_num_cases` cases are resolved once, from a single seed or from the config's
explicit `grad_u_location`/`coriolis_location`, and sliced into contiguous,
non-overlapping blocks per split, so train and every test dataset are disjoint by
construction. `num_time_steps` is scaled per split so that
`st_max / num_time_steps` (the St resolution) matches the training run.

Each split is run through the same pipeline as a single manual run: simulate
(scripts/launcher.py), postprocess (scripts/postprocessing.py), and build the
ML dataset (scripts/ml_postprocessing.py). Each batch is postprocessed in memory
as soon as it is simulated, so only one batch of spectra is held at a time.

Outputs, under results/<run.name from --config>_train/ and
results/<run.name>_test_st<St>/ for each test St:
  config.toml, structure_tensors.pkl, es_array.npy, the ml/ directory written by
  scripts/ml_postprocessing.py, and phi_batch_*.npy when --save_phi is set.
"""

import argparse
import logging
import tomllib
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import tomli_w

from scripts.launcher import simulate_ensemble_cases
from scripts.postprocessing import postprocess_arrays
from scripts.ml_postprocessing import main as build_ml_dataset
from src.sampler import DEFAULT_SAMPLER, resolve_case_parameters

logger = logging.getLogger(__name__)


def _run_name(base_name: str, suffix: str) -> str:
    """
    Builds run.name for one split.

    Args:
        base_name (str): run.name from the base launcher config.
        suffix (str): Split suffix, e.g. "train" or "test_st4.5".

    Returns:
        str: Run name with "." replaced by "p" for filesystem safety.
    """
    return f"{base_name}_{suffix}".replace(".", "p")


def _save_batches(
    phi_arrays: Iterable[np.ndarray], results_dir: Path
) -> Iterator[np.ndarray]:
    """
    Saves each batch as phi_batch_<i>.npy while passing it through.

    Args:
        phi_arrays (Iterable[np.ndarray]): Batches of spectra, each shape
            (batch, num_time_steps, 9, n_wavevectors).
        results_dir (Path): Directory to save the batches in.

    Yields:
        np.ndarray: Each batch, unchanged.
    """
    for i, phi_array in enumerate(phi_arrays):
        np.save(results_dir / f"phi_batch_{i}.npy", phi_array)
        yield phi_array


def _split_plan(
    st_train: float, st_test: list[float], train_num_cases: int, test_num_cases: int
) -> list[tuple[str, float, int]]:
    """
    Builds the ordered list of splits to run.

    Args:
        st_train (float): Training St_max.
        st_test (list[float]): Testing St_max values.
        train_num_cases (int): Number of cases in the training split.
        test_num_cases (int): Number of cases in each testing split.

    Returns:
        list[tuple[str, float, int]]: Per-split (suffix, st_max, num_cases),
            training split first.
    """
    plan = [("train", st_train, train_num_cases)]
    plan += [(f"test_st{st:g}", st, test_num_cases) for st in st_test]
    return plan


def main(
    config: Path,
    st_train: float,
    st_test: list[float],
    train_num_cases: int,
    test_num_cases: int,
    es_threshold: float,
    es_degree: int | None = None,
    save_phi: bool = False,
) -> None:
    """
    Builds train and test ML datasets from one shared sampled case pool.

    Args:
        config (Path): Base launcher toml config; supplies batch_size,
            use_coriolis, sd_degree, solver, evolve_k, seed, the optional
            grad_u_location/coriolis_location, and the reference num_time_steps.
        st_train (float): Training St_max.
        st_test (list[float]): Testing St_max values, one dataset each.
        train_num_cases (int): Number of cases in the training split.
        test_num_cases (int): Number of cases in each testing split.
        es_threshold (float): Threshold above which case is flagged as stopped.
        es_degree (int | None, optional): Highest spherical harmonic degree
            tested for early stopping. Defaults to `2 * (sd_degree // 4)`, the
            largest even degree the spherical design resolves reliably. Unused
            when the config's evolve_k is False, since the early-stopping
            check is skipped entirely in that case.
        save_phi (bool, optional): Whether to save each batch of spectra as
            phi_batch_<i>.npy. Defaults to False.

    Raises:
        ValueError: If the config does not describe an ensemble run, or if the
            resolved case pool is smaller than the splits require.
    """
    repo_root = Path(__file__).resolve().parents[1]

    with open(config, "rb") as f:
        base_config = tomllib.load(f)

    if not base_config["run_type"]["ensemble"]:
        raise ValueError("Config must describe an ensemble run.")

    base_name = base_config["run"]["name"]
    ensemble = base_config["ensemble"]
    params = base_config["params"]

    seed = ensemble["seed"]
    batch_size = ensemble["batch_size"]
    use_coriolis = ensemble["use_coriolis"]
    grad_u_location = ensemble.get("grad_u_location")
    coriolis_location = ensemble.get("coriolis_location")
    sampler = ensemble.get("sampler", DEFAULT_SAMPLER)
    sd_degree = params["sd_degree"]
    solver = params["solver"]
    evolve_k = params["evolve_k"]
    base_num_time_steps = params["num_time_steps"]

    if es_degree is None:
        es_degree = 2 * (sd_degree // 4)

    plan = _split_plan(st_train, st_test, train_num_cases, test_num_cases)
    total_num_samples = sum(num_cases for _, _, num_cases in plan)

    # resolve the full pool once; each split below takes a disjoint slice
    mean_velocity_gradients, coriolis_terms = resolve_case_parameters(
        total_num_samples,
        use_coriolis,
        seed=seed,
        grad_u_location=grad_u_location,
        coriolis_location=coriolis_location,
        sampler=sampler,
    )

    if mean_velocity_gradients.shape[0] < total_num_samples:
        raise ValueError(
            f"Resolved case pool holds {mean_velocity_gradients.shape[0]} cases, but "
            f"the splits need {total_num_samples}."
        )

    offset = 0
    for suffix, st_max, num_cases in plan:
        block_gradients = mean_velocity_gradients[offset : offset + num_cases]
        block_coriolis = (
            coriolis_terms[offset : offset + num_cases]
            if coriolis_terms is not None
            else None
        )
        num_time_steps = round(base_num_time_steps * st_max / st_train)

        run_name = _run_name(base_name, suffix)
        results_dir = repo_root / "results" / run_name
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Simulating split %s: %d cases, st_max=%g, num_time_steps=%d.",
            run_name,
            num_cases,
            st_max,
            num_time_steps,
        )

        phi_arrays = simulate_ensemble_cases(
            block_gradients,
            block_coriolis,
            num_time_steps,
            sd_degree,
            st_max=st_max,
            solver=solver,
            batch_size=batch_size,
            evolve_k=evolve_k,
        )
        if save_phi:
            phi_arrays = _save_batches(phi_arrays, results_dir)

        split_config = {
            "run": {"name": run_name},
            "run_type": {"ensemble": True},
            "params": {
                "num_time_steps": num_time_steps,
                "sd_degree": sd_degree,
                "st_max": st_max,
                "solver": solver,
                "evolve_k": evolve_k,
            },
            "ensemble": {
                "num_samples": num_cases,
                "batch_size": batch_size,
                "use_coriolis": use_coriolis,
                "seed": seed,
                "sampler": sampler,
                "case_offset": offset,
                "total_num_samples": total_num_samples,
            },
        }
        # carry explicit case locations so downstream steps resolve the same pool
        if grad_u_location is not None:
            split_config["ensemble"]["grad_u_location"] = grad_u_location
        if coriolis_location is not None:
            split_config["ensemble"]["coriolis_location"] = coriolis_location
        config_path = results_dir / "config.toml"
        with open(config_path, "wb") as f:
            tomli_w.dump(split_config, f)

        structure_tensors_path = results_dir / "structure_tensors.pkl"
        es_array_path = results_dir / "es_array.npy"
        postprocess_arrays(
            phi_arrays,
            es_array_path,
            es_degree,
            es_threshold,
            structure_tensors_output=structure_tensors_path,
            enforce_earlystopping=evolve_k,
        )
        build_ml_dataset(
            structure_tensors_path, es_array_path, config_path, results_dir / "ml"
        )

        offset += num_cases

    assert offset == total_num_samples, "Split offsets must cover the full sampled pool."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build train + n test ML datasets spanning different St ranges."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/launcher_config.toml"),
        help="Base launcher toml config for shared ensemble params.",
    )
    parser.add_argument(
        "--st_train", type=float, required=True, help="Training St_max."
    )
    parser.add_argument(
        "--st_test",
        type=float,
        nargs="*",
        default=[3, 4, 5],
        help="Testing St_max values, one dataset per value. Defaults to "
        "[3, 4, 5]; pass --st_test with no values for a train-only dataset.",
    )
    parser.add_argument(
        "--train_num_cases",
        type=int,
        required=True,
        help="Number of cases in the training split.",
    )
    parser.add_argument(
        "--test_num_cases",
        type=int,
        required=True,
        help="Number of cases in each testing split.",
    )
    parser.add_argument(
        "--es_threshold",
        type=float,
        default=1.6e-4,
        help="Threshold above which case is flagged as stopped.",
    )
    parser.add_argument(
        "--es_degree",
        type=int,
        default=None,
        help="Highest spherical harmonic degree tested for early stopping. "
        "Defaults to 2 * (sd_degree // 4).",
    )
    parser.add_argument(
        "--save_phi",
        action="store_true",
        help="Save each batch of spectra as phi_batch_<i>.npy.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(
        args.config,
        args.st_train,
        args.st_test,
        args.train_num_cases,
        args.test_num_cases,
        args.es_threshold,
        es_degree=args.es_degree,
        save_phi=args.save_phi,
    )
