"""
Validate the early stopping criterion on pure shear and AXE cases.

For each spherical design t-value in T_ARRAY:
  - Run the numerical solver
  - Find the stopping index via the early stopping criterion
  - Evaluate R, D, Q*, M, L, M6 at that time step (numerical and analytical)
  - Compute relative Frobenius errors against the t=325 analytical ground truth

Outputs:
  results/verify_es/verify_es_shear.csv
  results/verify_es/verify_es_axe.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytical_solutions import analytical_solution_pure_shear, phi_analytical_strain
from src.earlystopping import stopping_index
from src.rdt_solver import simulate_single
from src.structure_tensors import evaluate_M_6th_order_es
from src.tensor_utils import get_D_ij, get_L_ijpq, get_M_ijpq, get_Qs_ijk, get_R_ij, strain_rate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T_ARRAY        = [45, 87, 93, 109, 151]
T_GROUND       = 325
NUM_TIME_STEPS = 100
STMAX          = 10
OMEGA          = np.zeros(3)
SOLVER         = "rk4"
THR            = 1.5e-4#1.6e-4
ST_MAX = 4

l_max = lambda t: (t // 4) * 2


def _results_dir() -> Path:
    """
    Resolves and creates results directory for verify_es runs.

    Returns:
        Path: Path to rdt-sd/results/verify_es.
    """
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results" / "verify_es"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def pack_sol(ns):
    """(9, n, t) → (1, t, 9, n) for tensor_utils functions."""
    return ns.transpose(2, 0, 1)[np.newaxis]


def pack_sol_from_phi_tensor(phi_tensor, k_arr):
    """
    Convert analytical_solution_pure_shear output to sol format.

    phi_tensor : (t, n, 3, 3)
    k_arr      : (t, n, 3)
    Returns    : (1, t, 9, n)
    """
    # Pack phi into 6 components [phi_11, phi_22, phi_33, phi_12, phi_13, phi_23]
    phi6 = np.stack([
        phi_tensor[..., 0, 0],
        phi_tensor[..., 1, 1],
        phi_tensor[..., 2, 2],
        phi_tensor[..., 0, 1],
        phi_tensor[..., 0, 2],
        phi_tensor[..., 1, 2],
    ], axis=-1)                         # (t, n, 6)
    phi6 = phi6.transpose(2, 1, 0)     # (6, n, t)
    k3   = k_arr.transpose(2, 1, 0)   # (3, n, t)
    ns   = np.vstack([phi6, k3])       # (9, n, t)
    return pack_sol(ns)


def pack_sol_from_strain_anl(phi_anl, k_anl):
    """
    Convert phi_analytical_strain output to sol format.

    phi_anl : (t, 6, n)
    k_anl   : (t, 3, n)
    Returns : (1, t, 9, n)
    """
    phi6 = phi_anl.transpose(1, 2, 0)  # (6, n, t)
    k3   = k_anl.transpose(1, 2, 0)    # (3, n, t)
    ns   = np.vstack([phi6, k3])        # (9, n, t)
    return pack_sol(ns)


# ---------------------------------------------------------------------------
# Tensor evaluation
# ---------------------------------------------------------------------------

def compute_tensors_at(sol, stop_idx):
    """
    Evaluate all structure tensors at a single time step.

    sol      : (1, T, 9, n)
    stop_idx : int
    Returns  : dict with R, D, Qs, M, L, M6 — each squeezed to spatial dims only
    """
    s = sol[:, stop_idx:stop_idx + 1, :, :]  # (1, 1, 9, n)

    R  = get_R_ij(s)[..., 0, 0]    # (3, 3)
    D  = get_D_ij(s)[..., 0, 0]    # (3, 3)
    Qs = get_Qs_ijk(s)[..., 0, 0]  # (3, 3, 3)
    M  = get_M_ijpq(s)[..., 0, 0]  # (3, 3, 3, 3)
    L  = get_L_ijpq(s)[..., 0, 0]  # (3, 3, 3, 3)

    # M6 uses structure_tensors API: phi (6,n,t), k_arr (3,n,t)
    phi6 = sol[0, :, :6, :].transpose(1, 2, 0)  # (6, n, T)
    k3   = sol[0, :, 6:, :].transpose(1, 2, 0)  # (3, n, T)
    # Evaluate over only one time step to avoid computing the full (T,) array
    phi6_s = phi6[:, :, stop_idx:stop_idx + 1]
    k3_s   = k3[:, :, stop_idx:stop_idx + 1]
    M6 = evaluate_M_6th_order_es(phi6_s, k3_s)[..., 0]  # (3,3,3,3,3,3)

    return dict(R=R, D=D, Qs=Qs, M=M, L=L, M6=M6)


# ---------------------------------------------------------------------------
# Tensorially consistent error functions — same pattern as M_6_error()
# in earlystopping_thresholds.py, one per tensor order.
# All inputs are spatial-only (no time dimension); return a scalar.
# ---------------------------------------------------------------------------

def _tc_error(anl, num, spatial_idx):
    """
    Tensorially consistent scalar error.

    Matches the M_6_error() pattern in earlystopping_thresholds.py:
        numerator   = sqrt(einsum('...,...->')(diff, diff))
        denominator = sqrt(einsum('...,...->')(anl,  anl ))
    spatial_idx : einsum index string for the spatial dimensions only.
    """
    s = spatial_idx
    diff = anl - num
    numerator   = np.einsum(f'{s},{s}->', diff, diff) ** 0.5
    denominator = np.einsum(f'{s},{s}->', anl,  anl)  ** 0.5
    if numerator < 1e-10:
        return 0.0
    if denominator < 1e-10:   # tensor is theoretically zero (e.g. Q* for AXE)
        return float('nan')
    return float(numerator / denominator)


def R_error(anl, num):  return _tc_error(anl, num, 'ij')      # 2nd order
def D_error(anl, num):  return _tc_error(anl, num, 'ij')      # 2nd order
def Qs_error(anl, num): return _tc_error(anl, num, 'ijk')     # 3rd order
def M_error(anl, num):  return _tc_error(anl, num, 'ijpq')    # 4th order
def L_error(anl, num):  return _tc_error(anl, num, 'ijpq')    # 4th order
def M6_error(anl, num): return _tc_error(anl, num, 'ijpqkl')  # 6th order


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

def verify_one_case(get_ground_truth_sol, get_numerical_sol, case_name):
    """
    Run verification for one flow case.

    get_ground_truth_sol() → sol_ground (1, T, 9, n)
    get_numerical_sol(t)   → ns (9, n, T)
    """
    logger.info("=" * 60)
    logger.info("Case: %s.", case_name)
    logger.info("=" * 60)

    logger.info("Computing analytical ground truth (t=325).")
    sol_ground = get_ground_truth_sol()

    rows = []
    for t in T_ARRAY:
        logger.info("--- t = %d ---", t)
        ns = get_numerical_sol(t)

        deg = l_max(t)
        logger.info("Finding stopping index (l_max=%d).", deg)
        stop_idx = stopping_index(ns, degree=deg, thr=THR)
        if stop_idx is None:
            logger.warning("Stopping criterion never triggered for t=%d. Using last index.", t)
            stop_idx = NUM_TIME_STEPS - 1
        logger.info("Stopping index: %d.", stop_idx)

        sol_num = pack_sol(ns)

        tensors_anl = compute_tensors_at(sol_ground, stop_idx)
        tensors_num = compute_tensors_at(sol_num,    stop_idx)

        row = {
            "t_value":        t,
            "stopping_index": stop_idx,
            "error_R":        R_error( tensors_anl["R"],  tensors_num["R"]),
            "error_D":        D_error( tensors_anl["D"],  tensors_num["D"]),
            "error_Qs":       Qs_error(tensors_anl["Qs"], tensors_num["Qs"]),
            "error_M":        M_error( tensors_anl["M"],  tensors_num["M"]),
            "error_L":        L_error( tensors_anl["L"],  tensors_num["L"]),
            "error_M6":       M6_error(tensors_anl["M6"], tensors_num["M6"]),
        }
        rows.append(row)
        logger.info(
            "Errors: R=%.3e  D=%.3e  Qs=%.3e  M=%.3e  L=%.3e  M6=%.3e.",
            row["error_R"], row["error_D"], row["error_Qs"],
            row["error_M"], row["error_L"], row["error_M6"],
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

def _shear_G():
    return np.array([[1/2, 0, 1/2], [0, 0, 0], [-1/2, 0, -1/2]])


def _axe_G():
    return np.array([[-1, 0, 0], [0, 1/2, 0], [0, 0, 1/2]])


def _diag_transform():
    s = 1 / 2**0.5
    return np.array([[s, 0, -s], [0, 1, 0], [s, 0, s]])


def run_shear():
    G = _shear_G()
    tmax = ST_MAX / strain_rate(G[None, ...])[0]
    diag_transform = _diag_transform()

    def ground_truth():
        phi_anl, k_anl = analytical_solution_pure_shear(
            T_GROUND, tmax, NUM_TIME_STEPS, rot_matrix=diag_transform
        )
        return pack_sol_from_phi_tensor(phi_anl, k_anl)

    def numerical(t):
        sol, _ = simulate_single(G, tmax, NUM_TIME_STEPS, OMEGA, t, solver=SOLVER)
        return np.asarray(sol).transpose(1, 2, 0)  # (T, 9, n) -> (9, n, T)

    return verify_one_case(ground_truth, numerical, "pure_shear")


def run_axe():
    G = _axe_G()
    tmax = ST_MAX / strain_rate(G[None, ...])[0]

    def ground_truth():
        phi_anl, k_anl = phi_analytical_strain(
            T_GROUND, tmax, NUM_TIME_STEPS,
            s11=-1, s22=0.5, s33=0.5
        )
        return pack_sol_from_strain_anl(phi_anl, k_anl)

    def numerical(t):
        sol, _ = simulate_single(G, tmax, NUM_TIME_STEPS, OMEGA, t, solver=SOLVER)
        return np.asarray(sol).transpose(1, 2, 0)  # (T, 9, n) -> (9, n, T)

    return verify_one_case(ground_truth, numerical, "AXE")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Runs early-stopping verification for pure shear and AXE cases and saves results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results_dir = _results_dir()

    df_shear = run_shear()
    df_axe   = run_axe()

    shear_path = results_dir / "verify_es_shear.csv"
    axe_path   = results_dir / "verify_es_axe.csv"

    df_shear.to_csv(shear_path, index=False)
    df_axe.to_csv(axe_path,   index=False)

    logger.info("=" * 60)
    logger.info("Results saved:")
    logger.info("  %s", shear_path)
    logger.info("  %s", axe_path)

    logger.info("--- Pure Shear ---")
    logger.info("\n%s", df_shear.to_string(index=False))
    logger.info("--- AXE ---")
    logger.info("\n%s", df_axe.to_string(index=False))


if __name__ == "__main__":
    main()
