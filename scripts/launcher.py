"""Launcher script for simulating rdt velocity spectra or spectrum."""

import tomllib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.sampler import sampling_parameters, generate_mean_velocity_gradients, generate_coriolis_terms
from src.rdt_solver import simulate_parallel, simulate_single
from src.tensor_utils import strain_rate


def launch_parallel(
    num_samples: int,
    use_coriolis: bool,
    num_time_steps: int,
    sd_degree: int,
    st_max: float = 4,
    seed: int = 42,
    solver: str = "dopri5",
    batch_size: int = 64,
) -> list[np.ndarray]:
    """
    Launches RDT simulation for ensemble of sampled cases.

    Args:
        num_samples (int): Number of cases to sample and simulate.
        use_coriolis (bool): Whether to sample and apply Coriolis terms.
        num_time_steps (int): Number of time steps per case.
        sd_degree (int): Spherical design degree, sets initial spectrum and wavevectors.
        st_max (float, optional): Max strain time St = tmax * strain rate. Defaults to 4.
        seed (int, optional): Sobol sampler seed. Defaults to 42.
        solver (str, optional): Integration scheme, "dopri5" or "rk4". Defaults to "dopri5".
        batch_size (int, optional): Number of cases simulated per batch. Defaults to 64.

    Returns:
        list[np.ndarray]: Per-batch spectrum arrays, each shape
            (batch, num_time_steps, 9, n_wavevectors).
    """
    # sample mean velocity gradients and coriolis terms
    sampling_params = sampling_parameters(num_samples, 1, seed=seed)
    mean_velocity_gradients = generate_mean_velocity_gradients(sampling_params[:, :4])
    coriolis_terms = generate_coriolis_terms(sampling_params[:, 4:]) if use_coriolis else None
    S = strain_rate(mean_velocity_gradients)
    t_max_array = np.array([st_max / s for s in S])

    # evaluate over batches, do not save spectrum data
    phi_arrays = []

    for i in tqdm(range(0, num_samples, batch_size)):
        batch_mean_velocity_gradients = mean_velocity_gradients[i : i + batch_size]
        batch_coriolis_terms = coriolis_terms[i : i + batch_size] if use_coriolis else None
        batch_t_max_array = t_max_array[i : i + batch_size]

        phi_array, _ = simulate_parallel(
            batch_mean_velocity_gradients,
            batch_t_max_array,
            num_time_steps,
            batch_coriolis_terms,
            sd_degree,
            solver=solver,
        )

        phi_arrays.append(phi_array)

    return phi_arrays


def launch_rdt_single(
    grad_u: np.ndarray,
    omega: np.ndarray,
    num_time_steps: int,
    sd_degree: int,
    st_max: float = 4,
    solver: str = "dopri5",
) -> np.ndarray:
    """
    Launches RDT simulation for single velocity-gradient case.

    Args:
        grad_u (np.ndarray): Velocity-gradient tensor, shape (3, 3).
        omega (np.ndarray): Coriolis rotation vector, shape (3,).
        num_time_steps (int): Number of time steps.
        sd_degree (int): Spherical design degree, sets initial spectrum and wavevectors.
        st_max (float, optional): Max strain time St = tmax * strain rate. Defaults to 4.
        solver (str, optional): Integration scheme, "dopri5" or "rk4". Defaults to "dopri5".

    Returns:
        np.ndarray: Spectrum array, shape (num_time_steps, 9, n_wavevectors).
    """
    grad_u = np.asarray(grad_u)
    omega = np.asarray(omega)

    # derive tmax from st_max the same way as the ensemble path
    S = strain_rate(grad_u[None, ...])[0]
    tmax = st_max / S

    phi_array, _ = simulate_single(
        grad_u,
        tmax,
        num_time_steps,
        omega,
        sd_degree,
        solver=solver,
    )

    return phi_array


def main() -> None:
    """Loads launcher config, runs simulation, and saves resulting spectra."""
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "launcher_config.toml"

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    run_name = config["run"]["name"]
    is_ensemble = config["run_type"]["ensemble"]
    shared_params = config["params"]

    results_dir = repo_root / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # save config for reproducibility
    with open(results_dir / "config.toml", "wb") as f:
        tomllib.dump(config, f)

    # dispatch by run type and save results under configured run name
    if is_ensemble:
        phi_arrays = launch_parallel(**shared_params, **config["ensemble"])
        for i, phi_array in enumerate(phi_arrays):
            np.save(results_dir / f"phi_batch_{i}.npy", np.asarray(phi_array))
    else:
        single_params = dict(config["single"])
        single_params.setdefault("omega", [0.0, 0.0, 0.0])
        phi_array = launch_rdt_single(**single_params, **shared_params)
        np.save(results_dir / "phi_single.npy", np.asarray(phi_array))


if __name__ == "__main__":
    main()
