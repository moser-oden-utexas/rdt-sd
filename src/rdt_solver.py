"""Numerically solves rdt velocity spectrum using diffrax on jax."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve

from src.spherical_designs import init_wavenumbers_spherical_designs

jax.config.update("jax_enable_x64", True)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

def _unpack_phi(phi6: jnp.ndarray) -> jnp.ndarray:
    """Sends (6, ...) to (3, 3, ...)."""
    return jnp.array(
        [
            [phi6[0], phi6[3], phi6[4]],
            [phi6[3], phi6[1], phi6[5]],
            [phi6[4], phi6[5], phi6[2]],
        ]
    )


def _pack_phi(phi: jnp.ndarray) -> jnp.ndarray:
    """Sends (3, 3, ...) to (6, ...)."""
    return jnp.array([phi[0, 0], phi[1, 1], phi[2, 2], phi[0, 1], phi[0, 2], phi[1, 2]])


def _skew(omega: jnp.ndarray) -> jnp.ndarray:
    """Builds skew-symmetric W such that W v = omega x v for all v."""
    wx, wy, wz = omega
    return jnp.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])


def _project_div_free(y: jnp.ndarray) -> jnp.ndarray:
    """Projects phi to divergence-free subspace for each wavevector."""
    phi6, k = y[:6, :], y[6:, :]  # y (9, k)
    phi = _unpack_phi(phi6)  # (3, 3, k)
    kk = jnp.sum(k * k, axis=0)  # (k,)
    p = jnp.eye(3)[:, :, None] - k[None, :, :] * k[:, None, :] / kk[None, None, :]
    phi_proj = jnp.einsum("ilk, jmk, lmk -> ijk", p, p, phi)  # (3, 3, k)
    return jnp.vstack([_pack_phi(phi_proj), k])


def _rhs_tau_k(
    tau: float, y: jnp.ndarray, args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """Computes dy/dτ = tmax * dy/dt; y has shape (9,)."""
    # unpack arguments
    grad_u, omega, tmax = args  # (3, 3), (3,), float
    phi = _unpack_phi(y[:6])  # (3, 3)
    k = y[6:]
    kk = jnp.dot(k, k)

    # compute dphi/dτ
    dphi = -grad_u @ phi - phi @ grad_u.T
    dphi += 2.0 / kk * jnp.einsum("lk, i, l, kj -> ij", grad_u, k, k, phi)
    dphi += 2.0 / kk * jnp.einsum("lk, j, l, ik -> ij", grad_u, k, k, phi)

    w = _skew(omega)
    dphi += -2.0 * (w @ phi + phi @ w.T)

    # compute dk/dτ
    dk = -k @ grad_u

    rhs = jnp.concatenate([_pack_phi(dphi), dk])
    return tmax * rhs


def _rhs_tau(
    tau: float, y: jnp.ndarray, args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """Evaluates rhs per wavenumber and stacks back to (9, k)."""
    return jax.vmap(_rhs_tau_k, in_axes=(None, 1, None), out_axes=1)(tau, y, args)

def _load_y0(sd_degree: int) -> jnp.ndarray:
    """Builds initial state Y0 with shape (9, n_wavevectors)."""
    karr, _ = init_wavenumbers_spherical_designs(sd_degree)
    K0 = jnp.asarray(karr.T)
    K0K0 = jnp.sum(K0 * K0, axis=0)

    PHI0 = jnp.eye(3)[..., None] - jnp.einsum("in, jn -> ijn", K0, K0) / K0K0
    PHI0_6 = _pack_phi(PHI0)

    return jnp.vstack([PHI0_6, K0])


def _solve_single(
    Y0: jnp.ndarray,
    g_u: jnp.ndarray,
    om: jnp.ndarray,
    tm: jnp.ndarray,
    tau_eval: jnp.ndarray,
    solver: str,
) -> jnp.ndarray:
    """Integrates one case over tau_eval; returns shape (num_time_steps, 9, n_wavevectors)."""
    args = (g_u, om, tm)
    t0 = tau_eval[0]

    def step(carry, t_next):
        t_prev, y_prev = carry

        if solver == "dopri5":
            sol = diffeqsolve(
                ODETerm(_rhs_tau),
                Dopri5(),
                t0=t_prev,
                t1=t_next,
                dt0=None,
                y0=y_prev,
                args=args,
                stepsize_controller=PIDController(rtol=1e-6, atol=1e-8),
                saveat=SaveAt(t1=True),
            )
            y_next = sol.ys[-1]
        elif solver == "rk4":
            h = t_next - t_prev
            k1 = _rhs_tau(t_prev, y_prev, args)
            k2 = _rhs_tau(t_prev + 0.5 * h, y_prev + 0.5 * h * k1, args)
            k3 = _rhs_tau(t_prev + 0.5 * h, y_prev + 0.5 * h * k2, args)
            k4 = _rhs_tau(t_prev + h, y_prev + h * k3, args)
            y_next = y_prev + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y_next = _project_div_free(y_next)
        return (t_next, y_next), y_next

    _, ys = jax.lax.scan(step, init=(t0, Y0), xs=tau_eval[1:])
    return jnp.concatenate([Y0[None, ...], ys], axis=0)


@partial(jax.jit, static_argnums=(2, 4, 5))
def simulate_parallel(
    grad_u: np.ndarray | jnp.ndarray,
    tmax: np.ndarray | jnp.ndarray,
    num_time_steps: int = 100,
    omega: np.ndarray | jnp.ndarray | None = None,
    sd_degree: int = 109,
    solver: str = "dopri5",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulates RDT spectra for multiple cases in parallel with normalized time.

    Args:
        grad_u: shape (batch, 3, 3)
        tmax: shape (batch,)
        num_time_steps: number of time steps
        omega: None, shape (3,), or shape (batch, 3)
        sd_degree: spherical design degree, sets initial spectrum and wavevectors
        solver: integration scheme, "dopri5" (adaptive, diffrax Dopri5 +
            PIDController) or "rk4" (fixed-step classical RK4, one step
            per save interval). Defaults to "dopri5".

    Returns:
        sol: shape (batch, num_time_steps, 9, n_wavevectors)
        t_actual: shape (batch, num_time_steps)

    Raises:
        ValueError: if solver is not "dopri5" or "rk4".
    """
    if solver not in ("dopri5", "rk4"):
        raise ValueError(f'Unknown solver: {solver!r}. Expected "dopri5" or "rk4".')

    Y0 = _load_y0(sd_degree)

    grad_u_b = jnp.asarray(grad_u)
    tmax_b = jnp.asarray(tmax)
    batch = grad_u_b.shape[0]

    if omega is None:
        omega_b = jnp.zeros((batch, 3), dtype=grad_u_b.dtype)
    else:
        omega_arr = jnp.asarray(omega, dtype=grad_u_b.dtype)
        omega_b = (
            omega_arr[None, :].repeat(batch, axis=0)
            if omega_arr.ndim == 1
            else omega_arr
        )

    tau_eval = jnp.linspace(0.0, 1.0, num_time_steps)

    solve_single = lambda g_u, om, tm: _solve_single(Y0, g_u, om, tm, tau_eval, solver)
    sol = jax.vmap(solve_single, in_axes=(0, 0, 0), out_axes=0)(
        grad_u_b, omega_b, tmax_b
    )

    t_actual = tau_eval[None, :] * tmax_b[:, None]

    return sol, t_actual


@partial(jax.jit, static_argnums=(2, 4, 5))
def simulate_single(
    grad_u: np.ndarray | jnp.ndarray,
    tmax: np.ndarray | jnp.ndarray,
    num_time_steps: int = 100,
    omega: np.ndarray | jnp.ndarray | None = None,
    sd_degree: int = 109,
    solver: str = "dopri5",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulates RDT spectrum for a single case with normalized time.

    Args:
        grad_u: shape (3, 3)
        tmax: scalar
        num_time_steps: number of time steps
        omega: None or shape (3,)
        sd_degree: spherical design degree, sets initial spectrum and wavevectors
        solver: integration scheme, "dopri5" (adaptive, diffrax Dopri5 +
            PIDController) or "rk4" (fixed-step classical RK4, one step
            per save interval). Defaults to "dopri5".

    Returns:
        sol: shape (num_time_steps, 9, n_wavevectors)
        t_actual: shape (num_time_steps,)

    Raises:
        ValueError: if solver is not "dopri5" or "rk4".
    """
    if solver not in ("dopri5", "rk4"):
        raise ValueError(f'Unknown solver: {solver!r}. Expected "dopri5" or "rk4".')

    Y0 = _load_y0(sd_degree)

    grad_u_s = jnp.asarray(grad_u)
    tmax_s = jnp.asarray(tmax)
    omega_s = (
        jnp.zeros(3, dtype=grad_u_s.dtype)
        if omega is None
        else jnp.asarray(omega, dtype=grad_u_s.dtype)
    )

    tau_eval = jnp.linspace(0.0, 1.0, num_time_steps)

    sol = _solve_single(Y0, grad_u_s, omega_s, tmax_s, tau_eval, solver)
    t_actual = tau_eval * tmax_s

    return sol, t_actual