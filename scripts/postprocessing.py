"""
Computes structure tensors and early-stopping indices from saved phi arrays.

Reads one or more saved phi arrays (.npy, or .pkl for pickled ndarrays).
Multiple paths are treated as batch shards of one ensemble (e.g.
phi_batch_0.npy, phi_batch_1.npy, ...) and their results are concatenated
into a single combined output.

Outputs:
  <--structure_tensors_output>, a pickled dict from compute_structure_tensors.
  <--es_array_output>, a plain npy array of early-stopping indices.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.earlystopping import batched_stopping_index
from src.tensor_utils import compute_structure_tensors

logger = logging.getLogger(__name__)


def _load_phi_array(path: Path) -> np.ndarray:
    """
    Loads one phi array, adding batch axis if it is a single-case save.

    Args:
        path (Path): Path to saved phi array, shape (batch, time, 9, k) or
            (time, 9, k) for a single-case save. Pickled ndarrays (.pkl) and
            plain .npy files are both supported.

    Returns:
        np.ndarray: Phi array, shape (batch, time, 9, k).
    """
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            phi_array = pickle.load(f)
    else:
        phi_array = np.load(path)

    if phi_array.ndim == 3:
        phi_array = phi_array[None, ...]
    return phi_array


def _process_shard(
    phi_array: np.ndarray, es_degree: int, es_threshold: float, es_only: bool
) -> tuple[dict | None, np.ndarray]:
    """
    Computes structure tensors and early-stopping indices for one shard.

    Args:
        phi_array (np.ndarray): Phi array, shape (batch, time, 9, k).
        es_degree (int): Highest spherical harmonic degree tested for early stopping.
        es_threshold (float): Threshold above which case is flagged as stopped.
        es_only (bool): Skips structure tensor computation when True.

    Returns:
        tuple[dict | None, np.ndarray]: Structure tensors, each shape
            (..., batch, time), or None when es_only, and early-stopping
            index per case, shape (batch,).
    """
    structure_tensors = None if es_only else compute_structure_tensors(phi_array)
    es_array = batched_stopping_index(phi_array, degree=es_degree, thr=es_threshold)
    return structure_tensors, es_array


def _concatenate_shards(
    shard_results: list[tuple[dict | None, np.ndarray]],
) -> tuple[dict | None, np.ndarray]:
    """
    Concatenates per-shard structure tensors and es_arrays along batch axis.

    Args:
        shard_results (list[tuple[dict | None, np.ndarray]]): Per-shard
            structure tensors and early-stopping indices, as returned by
            _process_shard.

    Returns:
        tuple[dict | None, np.ndarray]: Combined structure tensors, each
            shape (..., batch, time), or None when shards carry no structure
            tensors, and combined early-stopping index per case, shape (batch,).
    """
    structure_tensors = None
    if shard_results[0][0] is not None:
        keys = shard_results[0][0].keys()
        structure_tensors = {
            key: np.concatenate([tensors[key] for tensors, _ in shard_results], axis=-2)
            for key in keys
        }

    es_array = np.concatenate([es_array for _, es_array in shard_results], axis=0)
    return structure_tensors, es_array


def main(
    phi_arrays: list[Path],
    es_array_output: Path,
    es_degree: int,
    es_threshold: float,
    structure_tensors_output: Path | None = None,
    es_only: bool = False,
) -> None:
    """
    Computes and saves structure tensors and early-stopping indices.

    Args:
        phi_arrays (list[Path]): Paths to one or more saved phi arrays. Multiple
            paths are treated as batch shards of one ensemble and concatenated.
        es_array_output (Path): Destination npy path for the combined es_array.
        es_degree (int): Highest spherical harmonic degree tested for early stopping.
        es_threshold (float): Threshold above which case is flagged as stopped.
        structure_tensors_output (Path | None, optional): Destination pkl path
            for the combined structure tensors dict. Ignored when es_only.
            Defaults to None.
        es_only (bool, optional): Skips structure tensor computation, only
            computing and saving es_array. Defaults to False.
    """
    shard_results = [
        _process_shard(_load_phi_array(path), es_degree, es_threshold, es_only)
        for path in tqdm(phi_arrays, desc="phi array shards")
    ]
    structure_tensors, es_array = _concatenate_shards(shard_results)

    es_array_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(es_array_output, es_array)
    logger.info("Saved es_array to %s.", es_array_output)

    if structure_tensors is not None:
        structure_tensors_output.parent.mkdir(parents=True, exist_ok=True)
        with open(structure_tensors_output, "wb") as f:
            pickle.dump(structure_tensors, f)
        logger.info("Saved structure tensors to %s.", structure_tensors_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute structure tensors and early-stopping indices from phi arrays."
    )
    parser.add_argument(
        "--phi_arrays",
        nargs="+",
        type=Path,
        required=True,
        help="Path(s) to saved phi arrays; multiple paths are batch shards of one ensemble.",
    )
    parser.add_argument(
        "--structure_tensors_output",
        type=Path,
        default=None,
        help="Destination pkl path for the combined structure tensors dict. Required unless --es_only.",
    )
    parser.add_argument(
        "--es_array_output",
        type=Path,
        required=True,
        help="Destination npy path for the combined es_array.",
    )
    parser.add_argument(
        "--es_degree",
        type=int,
        required=True,
        help="Highest spherical harmonic degree tested for early stopping.",
    )
    parser.add_argument(
        "--es_threshold",
        type=float,
        default=1.6e-4,
        help="Threshold above which case is flagged as stopped.",
    )
    parser.add_argument(
        "--es_only",
        action="store_true",
        help="Skip structure tensor computation; only compute and save es_array.",
    )
    args = parser.parse_args()

    if not args.es_only and args.structure_tensors_output is None:
        parser.error("--structure_tensors_output is required unless --es_only is set.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(
        args.phi_arrays,
        args.es_array_output,
        args.es_degree,
        args.es_threshold,
        structure_tensors_output=args.structure_tensors_output,
        es_only=args.es_only,
    )
