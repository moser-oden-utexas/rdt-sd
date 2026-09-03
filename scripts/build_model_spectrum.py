"""
Builds model spectrum from saved RDT phi arrays.

Blends the RDT-solved anisotropic spectrum with a closed-form isotropic
spectrum (src/tensor_utils.py::model_spectrum), then optionally computes
structure tensors and builds the ml/ dataset from the result, reusing the
source run's early-stopping indices instead of recomputing them.

Outputs, under results/<run_name>_model_spectrum/:
  phi_batch_*.npy and config.toml, always; structure_tensors.pkl, es_array.npy
  (copied from the source run), and the ml/ directory when --ml_postproc is set.
"""

import argparse
import logging
import pickle
import shutil
import tomllib
from pathlib import Path

import numpy as np

from scripts.ml_postprocessing import case_parameters_from_config
from scripts.ml_postprocessing import main as build_ml_dataset
from src.tensor_utils import compute_structure_tensors, model_spectrum

logger = logging.getLogger(__name__)


def _sorted_phi_paths(directory: Path) -> list[Path]:
    """
    Lists phi_batch_*.npy paths in a run directory, sorted by batch index.

    Args:
        directory (Path): Run directory to search.

    Returns:
        list[Path]: Sorted phi_batch_*.npy paths.
    """
    return sorted(
        directory.glob("phi_batch_*.npy"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])
    )


def _compute_structure_tensors(phi_paths: list[Path]) -> dict[str, np.ndarray]:
    """
    Computes and concatenates structure tensors across phi array shards.

    Args:
        phi_paths (list[Path]): Sorted phi_batch_*.npy paths, batch shards of
            one ensemble.

    Returns:
        dict[str, np.ndarray]: Structure tensors, each shape (..., num_cases,
        num_time_steps).
    """
    shard_tensors = [compute_structure_tensors(np.load(p)) for p in phi_paths]
    return {
        key: np.concatenate([tensors[key] for tensors in shard_tensors], axis=-2)
        for key in shard_tensors[0]
    }


def main(input_dir: Path, ml_postproc: bool = False) -> None:
    """
    Builds model-spectrum phi arrays from a run directory, optionally postprocessing them.

    Args:
        input_dir (Path): Run directory holding phi_batch_*.npy, config.toml,
            and (when ml_postproc) es_array.npy, e.g. results/pr_fluids_train.
        ml_postproc (bool, optional): Whether to compute structure tensors and
            build the ml/ dataset from the model spectrum. Reuses input_dir's
            es_array.npy rather than recomputing early-stopping indices.
            Defaults to False.

    Raises:
        ValueError: If config.toml is missing, does not describe an ensemble
            run, no phi_batch_*.npy files are found, or (when ml_postproc)
            es_array.npy is missing from input_dir.
    """
    config_path = input_dir / "config.toml"
    if not config_path.exists():
        raise ValueError(f"{config_path} does not exist.")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    if not config["run_type"]["ensemble"]:
        raise ValueError("Config must describe an ensemble run.")

    phi_paths = _sorted_phi_paths(input_dir)
    if not phi_paths:
        raise ValueError(f"No phi_batch_*.npy files found in {input_dir}.")

    es_array_path = input_dir / "es_array.npy"
    if ml_postproc and not es_array_path.exists():
        raise ValueError(
            f"{es_array_path} does not exist. --ml_postproc reuses the source "
            "run's early-stopping indices instead of recomputing them."
        )

    mean_velocity_gradients, _ = case_parameters_from_config(config_path)

    output_dir = input_dir.parent / f"{input_dir.name}_model_spectrum"
    output_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    for phi_path in phi_paths:
        sol = np.load(phi_path)
        batch = sol.shape[0]
        grad_u_batch = mean_velocity_gradients[offset : offset + batch]

        sol_model = model_spectrum(sol, grad_u_batch)
        np.save(output_dir / phi_path.name, sol_model)
        logger.info("Saved %s.", output_dir / phi_path.name)

        offset += batch

    shutil.copy2(config_path, output_dir / "config.toml")

    if ml_postproc:
        structure_tensors = _compute_structure_tensors(_sorted_phi_paths(output_dir))
        structure_tensors_path = output_dir / "structure_tensors.pkl"
        with open(structure_tensors_path, "wb") as f:
            pickle.dump(structure_tensors, f)
        logger.info("Saved structure tensors to %s.", structure_tensors_path)

        shutil.copy2(es_array_path, output_dir / "es_array.npy")
        logger.info("Reused early-stopping indices from %s.", es_array_path)

        build_ml_dataset(
            structure_tensors_path,
            output_dir / "es_array.npy",
            output_dir / "config.toml",
            output_dir / "ml",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build model spectrum from saved RDT phi arrays."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Run directory holding phi_batch_*.npy and config.toml.",
    )
    parser.add_argument(
        "--ml_postproc",
        action="store_true",
        help="Compute structure tensors and build the ml/ dataset from the model "
        "spectrum, reusing the source run's es_array.npy.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(args.input_dir, ml_postproc=args.ml_postproc)
