"""Thresholds for early stopping based on lmax energy fractions and M_6th_order errors."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.earlystopping import spectrum_over_time
from src.rdt_solver import simulate_single
from src.structure_tensors import evaluate_M_6th_order_es
from src.tensor_utils import strain_rate

plt.style.use("seaborn-v0_8-poster")

NUM_TIME_STEPS = 100
STMAX = 10
OMEGA = np.zeros(3)
SOLVER = "rk4"
G = np.array([[-1, 0, 0], [0, 0.5, 0], [0, 0.0, 0.5]])


def M_6_error(numerical: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Computes tensorally consistent error in order 6 M tensor.

    Args:
        numerical (np.ndarray): M tensor from case under test, shape (..., num_time_steps).
        predicted (np.ndarray): Reference M tensor, shape (..., num_time_steps).

    Returns:
        np.ndarray: Relative error per time step, shape (num_time_steps,).
    """
    diff = numerical - predicted
    numerator = np.einsum("ijpqklt, ijpqklt-> t", diff, diff) ** 0.5
    denominator = np.einsum("ijpqklt, ijpqklt-> t", numerical, numerical) ** 0.5
    error = numerator / denominator
    error[numerator < 1e-10] = 0
    return error


def _results_dir() -> Path:
    """
    Resolves and creates results directory for early stopping threshold runs.

    Returns:
        Path: Path to rdt-sd/results/es_thresholds.
    """
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results" / "es_thresholds"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _load_or_generate_case(t: int, t_max: float, results_dir: Path) -> np.ndarray:
    """
    Loads cached solver trajectory for degree t, or generates and caches it.

    Args:
        t (int): Spherical design degree.
        t_max (float): Max integration time.
        results_dir (Path): Directory to check for cache and save to.

    Returns:
        np.ndarray: Numerical solver state, shape (9, num_nodes, num_time_steps).
    """
    cache_path = results_dir / f"num_solver_t{t}.npy"
    if cache_path.exists():
        print(f"Loaded data for t: {t} from file.")
        return np.load(cache_path)

    sol, _ = simulate_single(G, t_max, NUM_TIME_STEPS, OMEGA, t, solver=SOLVER)
    # rdt-gen-shaped (9, num_nodes, num_time_steps), simulate_single returns (num_time_steps, 9, num_nodes)
    num_solver = np.asarray(sol).transpose(1, 2, 0)
    np.save(cache_path, num_solver)
    return num_solver


def _load_or_compute_spectrum(
    t: int, num_solver: np.ndarray, l_max: int, results_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads cached phi/kappa spectrum for degree t, or computes and caches it.

    Args:
        t (int): Spherical design degree.
        num_solver (np.ndarray): Numerical solver state, shape (9, num_nodes, num_time_steps).
        l_max (int): Highest spherical harmonic degree.
        results_dir (Path): Directory to check for cache and save to.

    Returns:
        tuple[np.ndarray, np.ndarray]: spectrum_phi and spectrum_kappa, each
            shape (num_time_steps, l_max+1).
    """
    phi_path = results_dir / f"spectrum_phi_t{t}_true.npy"
    kappa_path = results_dir / f"spectrum_kappa_t{t}_true.npy"

    if phi_path.exists() and kappa_path.exists():
        return np.load(phi_path), np.load(kappa_path)

    spectrum_phi, spectrum_kappa, _ = spectrum_over_time(num_solver, degree=l_max)
    spectrum_phi = np.array(spectrum_phi)
    spectrum_kappa = np.array(spectrum_kappa)
    np.save(phi_path, spectrum_phi)
    np.save(kappa_path, spectrum_kappa)
    return spectrum_phi, spectrum_kappa


def plot_sum_spectra(
    spectra_phi: list[np.ndarray],
    t_array: list[int],
    st_max: float,
    num_time_steps: int,
    results_dir: Path,
) -> None:
    """
    Plots energy fraction below l_max and its complement, over time, per degree.

    Saves two figures: sum_spectra.svg (energy fraction) and
    sum_spectra_comp.svg (1 - energy fraction, log scale).

    Args:
        spectra_phi (list[np.ndarray]): Per-degree phi spectra, each shape
            (num_time_steps, l_max+1).
        t_array (list[int]): Spherical design degrees, one per entry in spectra_phi.
        st_max (float): Max strain time St.
        num_time_steps (int): Number of time steps.
        results_dir (Path): Directory to save figures to.
    """
    x_val = np.linspace(0, st_max, num_time_steps)
    colors = "brgmc"

    # does the spectra components add upto 1?
    plt.figure(figsize=(6, 6))
    for i, t in enumerate(t_array):
        sum_phi = np.sum(spectra_phi[i], axis=1)
        plt.plot(x_val, sum_phi, "-" + colors[i], label=rf"$t = {t}$")
    plt.xlabel(r"$s$")
    plt.ylabel(r"$E_{\phi}^{<}/E_{\phi}$")
    plt.legend(ncol=2, loc="lower left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(results_dir / "sum_spectra.svg")
    plt.show()
    print(f"Plot saved at {results_dir / 'sum_spectra.svg'}")

    # complementary spectra: 1 - E^</E on log-linear scale
    plt.figure(figsize=(6, 6))
    for i, t in enumerate(t_array):
        sum_phi = np.sum(spectra_phi[i], axis=1)
        plt.semilogy(x_val, 1 - sum_phi, "-" + colors[i], label=rf"$t = {t}$")
    plt.xlabel(r"$s$")
    plt.ylabel(r"$1 - E_{\phi}^{<}/E_{\phi}$")
    plt.legend(ncol=2, loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(results_dir / "sum_spectra_comp.svg")
    plt.show()
    print(f"Plot saved at {results_dir / 'sum_spectra_comp.svg'}")


def plot_thresholds(
    spectra_phi: list[np.ndarray],
    M_errors: list[np.ndarray],
    t_array: list[int],
    l_max: callable,
    stopping_indices: list[int],
    st_max: float,
    num_time_steps: int,
    results_dir: Path,
) -> None:
    """
    Plots spectrum at l_max and M6 error over time, with stopping-index markers.

    Saves two figures: thresholds_1.svg (spectrum at l_max) and
    thresholds_2.svg (M6 error).

    Args:
        spectra_phi (list[np.ndarray]): Per-degree phi spectra, each shape
            (num_time_steps, l_max+1).
        M_errors (list[np.ndarray]): Per-degree M6 error over time.
        t_array (list[int]): Spherical design degrees.
        l_max (callable): Maps degree t to highest spherical harmonic degree tested.
        stopping_indices (list[int]): Per-degree stopping time index.
        st_max (float): Max strain time St.
        num_time_steps (int): Number of time steps.
        results_dir (Path): Directory to save figures to.
    """
    colors = "brgmc"
    x_vals = np.linspace(0, st_max, num_time_steps)

    fig1, ax1 = plt.subplots(figsize=(6, 6), constrained_layout=True)
    fig2, ax2 = plt.subplots(figsize=(6, 6), constrained_layout=True)

    for i, t in enumerate(t_array):
        spectrum_phi = spectra_phi[i]
        stop_ind = stopping_indices[i]
        spectrum_at_lmax = [spectrum_phi[step][l_max(t)] for step in range(len(spectrum_phi))]
        leg = rf"$\zeta$: {spectrum_at_lmax[stop_ind]:.2e}"

        ax1.semilogy(x_vals, spectrum_at_lmax, "-" + colors[i])
        ax1.semilogy(
            x_vals[stop_ind],
            spectrum_at_lmax[stop_ind],
            color=colors[i],
            linestyle="-",
            marker="*",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
            label=leg,
        )
        print(f"t: {t}, stopping index: {stop_ind}, spectrum at lmax: {spectrum_at_lmax[stop_ind]:.4e}")

        ax2.semilogy(x_vals, M_errors[i], "-" + colors[i], label=f"t: {t}")
        ax2.semilogy(x_vals[stop_ind], M_errors[i][stop_ind], "k*")
        print(f"t: {t}, stopping index: {stop_ind}, M6 error: {M_errors[i][stop_ind]:.4e}")

    ax1.set_xlabel(r"$s$")
    ax1.set_ylabel(r"$E^{l_m}_{\phi}/E_{\phi}$")
    ax1.set_ylim([1e-6, 1])
    ax1.grid()

    ax2.set_xlabel(r"$s$")
    ax2.legend()
    ax2.set_ylabel(r"$\mathcal{E}_N$")
    ax2.grid()
    ax2.set_ylim([1e-6, 1])

    fig1.savefig(results_dir / "thresholds_1.svg")
    fig1.show()
    fig2.savefig(results_dir / "thresholds_2.svg")
    fig2.show()
    print(f"Plots saved at {results_dir / 'thresholds_1.svg'} and {results_dir / 'thresholds_2.svg'}")


def main() -> None:
    """Determines early-stopping thresholds by comparing spectral/M6 convergence across degrees."""
    results_dir = _results_dir()

    t_array = [45, 87, 93, 109, 151]
    t_reference = 325
    l_max = lambda t: (t // 4) * 2  # closest even mode

    t_max = STMAX / strain_rate(G[None, ...])[0]
    print("TMAX = ", t_max)

    spectra_phi = []
    M_arrays = []

    for t in t_array + [t_reference]:
        print(f"l_max - {l_max(t)} for t - {t}")

        num_solver = _load_or_generate_case(t, t_max, results_dir)

        if t != t_reference:
            spectrum_phi, _ = _load_or_compute_spectrum(t, num_solver, l_max(t), results_dir)
            spectra_phi.append(spectrum_phi)

        M_order6 = evaluate_M_6th_order_es(num_solver[:6], num_solver[6:])
        M_arrays.append(M_order6)

    plot_sum_spectra(spectra_phi, t_array, STMAX, NUM_TIME_STEPS, results_dir)

    M_errors = [M_6_error(M_arrays[i], M_arrays[-1]) for i in range(len(t_array))]
    stopping_indices = []

    print("t, time index for dM < 1e-3")
    for i, t in enumerate(t_array):
        t_ind = np.where(M_errors[i] > 1e-3)[0][0] - 1
        print(f"{t}: {t_ind}")
        stopping_indices.append(t_ind)

    plot_thresholds(
        spectra_phi, M_errors, t_array, l_max, stopping_indices, STMAX, NUM_TIME_STEPS, results_dir
    )


if __name__ == "__main__":
    main()
