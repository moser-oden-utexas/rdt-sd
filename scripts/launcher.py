"""Launcher script for simulating rdt velocity spectra or spectrum."""

import tomllib
from pathlib import Path

import numpy as np
import tomli_w
from tqdm import tqdm

from src.sampler import resolve_case_parameters
from src.rdt_solver import (
    initial_state,
    simulate_from_state,
    simulate_parallel,
    simulate_single,
)
from src.tensor_utils import strain_rate


def simulate_ensemble_cases(
    mean_velocity_gradients: np.ndarray,
    coriolis_terms: np.ndarray | None,
    num_time_steps: int,
    sd_degree: int,
    st_max: float = 4,
    solver: str = "dopri5",
    batch_size: int = 64,
    evolve_k: bool = True,
) -> list[np.ndarray]:
    """
    Integrates given cases in batches.

    Args:
        mean_velocity_gradients (np.ndarray): Mean velocity gradients, shape
            (num_cases, 3, 3).
        coriolis_terms (np.ndarray | None): Coriolis terms, shape
            (num_cases, 3), or None to run without rotation.
        num_time_steps (int): Number of time steps per case.
        sd_degree (int): Spherical design degree, sets initial spectrum and wavevectors.
        st_max (float, optional): Max strain time St = tmax * strain rate. Defaults to 4.
        solver (str, optional): Integration scheme, "dopri5" or "rk4". Defaults to "dopri5".
        batch_size (int, optional): Number of cases simulated per batch. Defaults to 64.
        evolve_k (bool, optional): Whether wavevectors evolve under the mean
            velocity gradient. Defaults to True.

    Returns:
        list[np.ndarray]: Per-batch spectrum arrays, each shape
            (batch, num_time_steps, 9, n_wavevectors).
    """
    num_cases = mean_velocity_gradients.shape[0]
    S = strain_rate(mean_velocity_gradients)
    t_max_array = np.array([st_max / s for s in S])

    # evaluate over batches, do not save spectrum data
    phi_arrays = []

    for i in tqdm(range(0, num_cases, batch_size)):
        batch_mean_velocity_gradients = mean_velocity_gradients[i : i + batch_size]
        batch_coriolis_terms = (
            coriolis_terms[i : i + batch_size] if coriolis_terms is not None else None
        )
        batch_t_max_array = t_max_array[i : i + batch_size]

        phi_array, _ = simulate_parallel(
            batch_mean_velocity_gradients,
            batch_t_max_array,
            num_time_steps,
            batch_coriolis_terms,
            sd_degree,
            solver=solver,
            evolve_k=evolve_k,
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
    evolve_k: bool = True,
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
        evolve_k (bool, optional): Whether wavevectors evolve under the mean
            velocity gradient. Defaults to True.

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
        evolve_k=evolve_k,
    )

    return phi_array


def launch_rdt_stages(
    grad_u: np.ndarray,
    omega: np.ndarray,
    num_time_steps: int,
    sd_degree: int,
    st_max: np.ndarray,
    solver: str = "dopri5",
    evolve_k: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Launches RDT simulation for single case under piecewise-constant gradients.

    Stage i applies grad_u[i] for st_max[i] of its own strain time, starting
    from the state stage i-1 ended on. Snapshots are taken on one globally
    uniform St grid spanning 0 to sum(st_max), so the returned array has the
    same layout as a plain single-case run.

    Args:
        grad_u (np.ndarray): Per-stage velocity-gradient tensors, shape
            (num_stages, 3, 3).
        omega (np.ndarray): Coriolis rotation vector, shape (3,), shared by all stages.
        num_time_steps (int): Total number of time steps across all stages.
        sd_degree (int): Spherical design degree, sets initial spectrum and wavevectors.
        st_max (np.ndarray): Per-stage strain-time durations, shape (num_stages,).
            Stage i spans St = sum(st_max[:i]) to sum(st_max[:i + 1]).
        solver (str, optional): Integration scheme, "dopri5" or "rk4". Defaults to "dopri5".
        evolve_k (bool, optional): Whether wavevectors evolve under the mean
            velocity gradient. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: Spectrum array, shape (num_time_steps,
            9, n_wavevectors), and its St axis, shape (num_time_steps,).

    Raises:
        ValueError: If grad_u and st_max disagree on the number of stages, or
            if any stage has a non-positive strain-time duration.
    """
    grad_u = np.asarray(grad_u, dtype=float)
    omega = np.asarray(omega, dtype=float)
    spans = np.asarray(st_max, dtype=float)

    if grad_u.ndim != 3 or grad_u.shape[1:] != (3, 3):
        raise ValueError(
            f"Expected grad_u of shape (num_stages, 3, 3), got {grad_u.shape}."
        )
    if spans.ndim != 1 or spans.shape[0] != grad_u.shape[0]:
        raise ValueError(
            f"Expected st_max of shape ({grad_u.shape[0]},) to match grad_u, "
            f"got {spans.shape}."
        )
    if not np.all(spans > 0):
        raise ValueError(f"Every stage needs a positive st_max, got {spans.tolist()}.")

    # global uniform St grid, with each point assigned to the stage it falls in.
    # points landing exactly on an interior breakpoint go to the earlier stage.
    breaks = np.concatenate([[0.0], np.cumsum(spans)])
    st_axis = np.linspace(0.0, breaks[-1], num_time_steps)
    stage_of = np.searchsorted(breaks[1:-1], st_axis, side="left")

    Y = initial_state(sd_degree)
    phi_stages = []

    for i, g_u in enumerate(grad_u):
        # normalized times this stage owns, plus the 0 and 1 endpoints needed
        # to start from the carried state and hand off exactly at the breakpoint
        local_tau = (st_axis[stage_of == i] - breaks[i]) / spans[i]
        nodes = np.unique(np.concatenate([[0.0], local_tau, [1.0]]))
        saved = np.isin(nodes, local_tau)

        # derive tmax from this stage's own strain rate, as the single-case path does
        tmax = spans[i] / strain_rate(g_u[None, ...])[0]

        sol, _ = simulate_from_state(
            Y, g_u, tmax, nodes, omega, solver=solver, evolve_k=evolve_k
        )
        sol = np.asarray(sol)

        phi_stages.append(sol[saved])
        Y = sol[-1]

    phi_array = np.concatenate(phi_stages, axis=0)
    assert (
        phi_array.shape[0] == num_time_steps
    ), "Stage snapshots must partition St grid."

    return phi_array, st_axis


def main() -> None:
    """Loads launcher config, runs simulation, and saves resulting spectra."""
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "launcher_config.toml"

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    run_name = config["run"]["name"]
    is_ensemble = config["run_type"]["ensemble"]
    is_stages = config["run_type"].get("stages", False)
    shared_params = config["params"]

    if is_ensemble and is_stages:
        raise ValueError("Run type must be either ensemble or stages, not both.")

    results_dir = repo_root / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # resolve ensemble cases before saving config, so the saved num_samples
    # reflects the pool actually simulated when cases come from explicit files
    if is_ensemble:
        ensemble = config["ensemble"]
        mean_velocity_gradients, coriolis_terms = resolve_case_parameters(
            ensemble["num_samples"],
            ensemble["use_coriolis"],
            seed=ensemble["seed"],
            grad_u_location=ensemble.get("grad_u_location"),
            coriolis_location=ensemble.get("coriolis_location"),
        )
        ensemble["num_samples"] = mean_velocity_gradients.shape[0]

    # save config for reproducibility
    with open(results_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # dispatch by run type and save results under configured run name
    if is_ensemble:
        phi_arrays = simulate_ensemble_cases(
            mean_velocity_gradients,
            coriolis_terms,
            batch_size=ensemble["batch_size"],
            **shared_params,
        )
        for i, phi_array in enumerate(phi_arrays):
            np.save(results_dir / f"phi_batch_{i}.npy", np.asarray(phi_array))
    elif is_stages:
        # st_max comes from [stages], so params are passed explicitly rather
        # than splatted; [params].st_max is unused in this mode
        stages = config["stages"]
        phi_array, st_axis = launch_rdt_stages(
            stages["grad_u"],
            stages.get("omega", [0.0, 0.0, 0.0]),
            shared_params["num_time_steps"],
            shared_params["sd_degree"],
            stages["st_max"],
            solver=shared_params["solver"],
            evolve_k=shared_params["evolve_k"],
        )
        np.save(results_dir / "phi_single.npy", phi_array)
        np.save(results_dir / "st_axis.npy", st_axis)
        np.save(
            results_dir / "kappa_init.npy",
            np.asarray(initial_state(shared_params["sd_degree"])[6:]),
        )
    else:
        single_params = dict(config["single"])
        single_params.setdefault("omega", [0.0, 0.0, 0.0])
        phi_array = launch_rdt_single(**single_params, **shared_params)
        np.save(results_dir / "phi_single.npy", np.asarray(phi_array))


if __name__ == "__main__":
    main()
