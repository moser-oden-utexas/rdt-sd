"""Provides tensor helpers for rdt simulation and downstream workflows."""

from __future__ import annotations

import itertools
from itertools import permutations

import numpy as np
from einops import rearrange

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


def get_array_module(arr):
    """Get numpy or cupy module based on array type."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


EPS = np.zeros((3, 3, 3), dtype=float)
EPS[0, 1, 2] = EPS[1, 2, 0] = EPS[2, 0, 1] = 1.0
EPS[0, 2, 1] = EPS[2, 1, 0] = EPS[1, 0, 2] = -1.0


def flatten_q(q):
    """
    Flattens q tensor to 27 components.

    Args:
        q: q tensor with shape (..., 3, 3, 3).

    Returns:
        Array with shape (..., 27).

    Raises:
        ValueError: If trailing shape is not (3, 3, 3).
    """
    if q.shape[-3:] != (3, 3, 3):
        raise ValueError(f"Expected trailing shape (3, 3, 3), got {q.shape}.")
    leading = q.shape[:-3]
    return q.reshape(leading + (27,))


def unflatten_q(q_flat):
    """
    Unflattens 27 components to q tensor.

    Args:
        q_flat: Flattened tensor with shape (..., 27).

    Returns:
        Array with shape (..., 3, 3, 3).

    Raises:
        ValueError: If trailing shape is not (27,).
    """
    if q_flat.shape[-1] != 27:
        raise ValueError(f"Expected trailing shape (27,), got {q_flat.shape}.")
    leading = q_flat.shape[:-1]
    return q_flat.reshape(leading + (3, 3, 3))


def strain_rate(G: np.ndarray) -> np.ndarray:
    """
    Computes strain-rate magnitude for batch of velocity-gradient tensors.

    Args:
        G (np.ndarray): Velocity-gradient tensors, shape (batch, 3, 3).

    Returns:
        np.ndarray: Strain-rate magnitude per case, shape (batch,).
    """
    S = 0.5 * (G + G.transpose(0, 2, 1))
    SijSij = np.einsum("nij, nij -> n", S, S)
    return (2 * SijSij) ** 0.5


def symmetric_part(q: np.ndarray) -> np.ndarray:
    """
    Computes fully symmetric part of q tensor.

    Args:
        q (np.ndarray): q tensor with shape (..., 3, 3, 3).

    Returns:
        np.ndarray: Fully symmetric tensor with same shape as input.
    """
    if q.shape[-3:] != (3, 3, 3):
        raise ValueError(f"Expected trailing shape (3, 3, 3), got {q.shape}.")

    base_axes = list(range(q.ndim - 3))
    out = np.zeros_like(q)
    for perm in itertools.permutations([0, 1, 2]):
        axes = base_axes + [q.ndim - 3 + p for p in perm]
        out = out + np.transpose(q, axes=axes)
    return out / 6.0


def derive_tensors_from_q(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derives r, d, and qs tensors from q.

    Args:
        q (np.ndarray): q tensor with shape (..., 3, 3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: r with shape (..., 3, 3), d with shape
        (..., 3, 3), and qs with shape (..., 3, 3, 3).
    """
    if q.shape[-3:] != (3, 3, 3):
        raise ValueError(f"Expected trailing shape (3, 3, 3), got {q.shape}.")

    r = np.einsum("imp,...mjp->...ij", EPS, q)
    d = np.einsum("imp,...pmj->...ij", EPS, q)
    qs = symmetric_part(q)
    return r, d, qs


def split_solution(sol):
    """From sol (b, t, 9, k) to phi (3, 3, b, t, k) and k (3, b, t, k)."""
    xp = get_array_module(sol)
    b, t, _, k = sol.shape
    # reconstruct phi
    phi = xp.zeros([3, 3, b, t, k], dtype=sol.dtype)
    phi[0, 0] = sol[:, :, 0, :]
    phi[1, 1] = sol[:, :, 1, :]
    phi[2, 2] = sol[:, :, 2, :]
    phi[0, 1] = sol[:, :, 3, :]
    phi[0, 2] = sol[:, :, 4, :]
    phi[1, 2] = sol[:, :, 5, :]
    # add lower triangle using Hermitian symmetry
    phi[1, 0] = xp.conj(phi[0, 1])
    phi[2, 0] = xp.conj(phi[0, 2])
    phi[2, 1] = xp.conj(phi[1, 2])

    # extract wavenumber vector
    k = sol[:, :, 6:, :]
    k = rearrange(k, "b t i k -> i b t k")

    return phi, k


def get_R_ij(sol):
    """From sol (b, t, 9, k) to R_ij (3, 3, b, t)."""
    xp = get_array_module(sol)
    C = 0.15

    phi, _ = split_solution(sol)
    N = phi.shape[-1]  # number of wavenumbers

    integral = xp.sum(phi, axis=-1)  # (i, j, b, t)
    integral *= 4 * xp.pi / N

    return C * integral


def get_q2(sol):
    """From sol (b, t, 9, k) to q2 (b, t)."""
    xp = get_array_module(sol)
    R_ij = get_R_ij(sol)
    q2 = xp.trace(R_ij, axis1=0, axis2=1)
    return q2


def get_D_ij(sol):
    """From sol (b, t, 9, k) to D_ij (3, 3, b, t)."""
    xp = get_array_module(sol)
    C = 0.15

    phi, k = split_solution(sol)
    N = k.shape[-1]  # number of wavenumbers

    # compute k² = |k|² for each wavenumber
    # k: (3, b, t, k)
    kk = xp.sum(k**2, axis=0)  # shape: (b, t, k)

    # compute sum_k (Φ_nn * k_i * k_j / k²)
    integral = xp.einsum("nnbtk, ibtk, jbtk, btk -> ijbt", phi, k, k, 1.0 / kk)
    integral *= 4 * xp.pi / N

    return C * integral


def get_Q_ijk(sol):
    """From sol (b, t, 9, k) to Q_ijk (3, 3, 3, b, t)."""
    xp = get_array_module(sol)
    C = 0.15

    phi, k = split_solution(sol)
    N = k.shape[-1]  # number of wavenumbers

    # compute k² = |k|² for each wavenumber
    # k: (3, b, t, k)
    kk = xp.sum(k**2, axis=0)  # shape: (b, t, k)

    # convert EPS to same array type
    eps = xp.asarray(EPS)

    # compute sum_k e_ipq * (Φ_jq * k_p * k_r / k²)
    integral = xp.einsum(
        "ipq, jqbtk, pbtk, rbtk, btk -> ijrbt", eps, phi, k, k, 1.0 / kk
    )
    integral *= 4 * xp.pi / N

    return C * integral


def fully_symmetrize_tensor(tensor):
    """Fully symmetrize a tensor, assuming last two indices are batch, time."""
    xp = get_array_module(tensor)
    dim = tensor.ndim - 2
    perms = list(permutations(range(dim)))
    return sum(xp.transpose(tensor, p + (dim, dim + 1)) for p in perms) / len(perms)


def get_Qs_ijk(sol):
    """From sol (b, t, 9, k) to Qs_ijk (3, 3, 3, b, t)."""
    Q_ijk = get_Q_ijk(sol)
    Qs_ijk = fully_symmetrize_tensor(Q_ijk)
    return Qs_ijk


def get_M_ijpq(sol):
    """From sol (b, t, 9, k) to M_ijpq (3, 3, 3, 3, b, t)."""
    xp = get_array_module(sol)
    C = 0.15

    phi, k = split_solution(sol)
    N = k.shape[-1]  # number of wavenumbers

    # compute k² = |k|² for each wavenumber
    # k: (3, b, t, k)
    kk = xp.sum(k**2, axis=0)  # shape: (b, t, k)

    # compute sum_k (Φ_ij * k_p * k_q / k²)
    integral = xp.einsum("ijbtk, pbtk, qbtk, btk -> ijpqbt", phi, k, k, 1.0 / kk)
    integral *= 4 * xp.pi / N

    return C * integral


def get_Ms_ijpq(sol):
    """From sol (b, t, 9, k) to Ms_ijpq (3, 3, 3, 3, b, t)."""
    M_ijpq = get_M_ijpq(sol)
    Ms_ijpq = fully_symmetrize_tensor(M_ijpq)
    return Ms_ijpq


def get_L_ijpq(sol):
    """From sol (b, t, 9, k) to L_ijpq (3, 3, 3, 3, b, t)."""
    xp = get_array_module(sol)
    C = 0.15

    phi, k = split_solution(sol)
    N = k.shape[-1]

    # compute k⁴ = |k|⁴ for each wavenumber
    # k: (3, b, t, k)
    kk = xp.sum(k**2, axis=0)  # shape: (b, t, k)
    kkkk = kk**2

    # compute sum_k (Φ_nn * k_i * k_j * k_p * k_q / k⁴)
    integral = xp.einsum(
        "nnbtk, ibtk, jbtk, pbtk, qbtk, btk -> ijpqbt", phi, k, k, k, k, 1.0 / kkkk
    )
    integral *= 4 * xp.pi / N

    return C * integral


def get_J_ijrpq(sol):
    """From sol (b, t, 9, k) to J_ijrpq (3, 3, 3, 3, 3, b, t)."""
    xp = get_array_module(sol)
    C = 0.15

    phi, k = split_solution(sol)
    N = k.shape[-1]

    # compute k⁴ = |k|⁴ for each wavenumber
    # k: (3, b, t, k)
    kk = xp.sum(k**2, axis=0)  # shape: (b, t, k)
    kkkk = kk**2

    # convert EPS to same array type
    eps = xp.asarray(EPS)

    # compute sum_k e_ins * (Φ_sj * k_n * k_r * k_p * k_q / k⁴)
    integral = xp.einsum(
        "ins, sjbtk, nbtk, rbtk, pbtk, qbtk, btk -> ijrpqbt",
        eps,
        phi,
        k,
        k,
        k,
        k,
        1.0 / kkkk,
    )
    integral *= 4 * xp.pi / N

    return C * integral


def get_anisotropy(A_ij):
    """
    From A_ij (3, 3, b, t) to b_ij (b, t, 3, 3).
    A_ij is any second-order tensor.
    """
    xp = get_array_module(A_ij)

    # transpose to shape (b, t, 3, 3)
    A_ij = rearrange(A_ij, "i j b t -> b t i j")

    trace = xp.trace(A_ij, axis1=-2, axis2=-1)
    trace = rearrange(trace, "b t -> b t 1 1")

    # compute b_ij = A_ij / trace(A_ij) - δ_ij/3
    identity = xp.eye(3)
    b_ij = A_ij / trace - identity / 3

    return b_ij


def flatten_batch_time(tensor):
    """From tensor (..., b, t) to tensor (b*t, ...)."""
    xp = get_array_module(tensor)
    n_leading = tensor.ndim - 2

    perm = list(range(n_leading, tensor.ndim)) + list(range(n_leading))
    tensor_transposed = xp.transpose(tensor, perm)

    b, t = tensor_transposed.shape[:2]
    new_shape = (b * t,) + tensor_transposed.shape[2:]

    return tensor_transposed.reshape(new_shape)


TENSOR_KEYS = (
    "q2",
    "r_ij",
    "d_ij",
    "q_ijk",
    "qs_ijk",
    "m_ijpq",
    "ms_ijpq",
    "l_ijpq",
    "j_ijrpq",
)


def compute_normalized_tensors(sol_subset):
    """
    Computes normalized tensors for sol subset.

    Args:
        sol_subset: Solution subset with shape (batch, time, 9, k). Can be numpy or cupy.

    Returns:
        tuple: (q2_flat, tensors_norm) where q2_flat has shape (batch*time,) and tensors_norm
        maps tensor names to arrays with shape (..., batch, time).
    """
    q2 = get_q2(sol_subset) 
    tensors = {
        "r_ij": get_R_ij(sol_subset),
        "d_ij": get_D_ij(sol_subset),
        "q_ijk": get_Q_ijk(sol_subset),
        "qs_ijk": get_Qs_ijk(sol_subset),
        "m_ijpq": get_M_ijpq(sol_subset),
        "ms_ijpq": get_Ms_ijpq(sol_subset),
        "l_ijpq": get_L_ijpq(sol_subset),
        "j_ijrpq": get_J_ijrpq(sol_subset)
    }
    tensors_norm = {k: v / q2[None, None, :, :] for k, v in tensors.items()}
    tensors_norm["q2"] = q2 
    return tensors_norm

def compute_structure_tensors(sol_subset):
    """
    Computes structure tensors for sol subset.

    Args:
        sol_subset: Solution subset with shape (batch, time, 9, k). Can be numpy or cupy.

    Returns:
        dict: Maps tensor names to arrays with shape (..., batch, time).
    """

    tensors = {
        "R_ij": get_R_ij(sol_subset),
        "D_ij": get_D_ij(sol_subset),
        "Q_ijk": get_Q_ijk(sol_subset),
        "Qs_ijk": get_Qs_ijk(sol_subset),
        "M_ijpq": get_M_ijpq(sol_subset),
        "Ms_ijpq": get_Ms_ijpq(sol_subset),
        "L_ijpq": get_L_ijpq(sol_subset),
        "J_ijrpq": get_J_ijrpq(sol_subset),
        "q2": get_q2(sol_subset) 
    }

    return tensors


def tensor_bt_to_front(tensor_np: np.ndarray) -> np.ndarray:
    """
    Moves batch and time axes to front and squeezes batch if it is 1.

    Args:
        tensor_np (np.ndarray): Tensor with shape (..., b, t) where b and t are last dims.

    Returns:
        np.ndarray: Tensor with shape (t, ...) if b==1 else (b, t, ...).
    """
    n_dims = tensor_np.ndim
    axes = [-2, -1] + list(range(n_dims - 2))
    out = np.transpose(tensor_np, axes)
    if out.shape[0] == 1:
        out = out.squeeze(0)
    return out
