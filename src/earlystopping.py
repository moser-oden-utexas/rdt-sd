"""Helpers to evaluate stopping criteria for deciding valid rdt time steps."""

import numpy as np
import sphericart
from tqdm import tqdm


def psd_total_energy(data: np.ndarray) -> float:
    """
    Evaluates total energy from power spectral density.

    Args:
        data (np.ndarray): Spectrum data, shape (*, num_nodes).

    Returns:
        float: Total energy.
    """
    num_nodes = data.shape[-1]
    return 4 * np.pi / num_nodes * np.sum(data * np.conj(data))


def ps(data: np.ndarray, Y: np.ndarray, degree: int) -> float:
    """
    Computes spherical-harmonic-decomposed power spectrum coefficient at degree.

    Uses sphericart's real spherical harmonics basis, so conjugation is
    unnecessary (kept implicitly by real arithmetic).

    Args:
        data (np.ndarray): Field values at each node.
        Y (np.ndarray): Precomputed spherical harmonics, shape (num_nodes, (l_max+1)^2).
        degree (int): Spherical harmonic degree to evaluate.

    Returns:
        float: Power spectrum coefficient at degree.
    """
    num_nodes = data.size
    spectrum_coef = 0.0
    for m in range(-degree, degree + 1):
        idx = degree * degree + degree + m
        mode_lm = Y[:, idx].reshape(data.shape)
        coef_lm = 4.0 * np.pi / num_nodes * np.sum(mode_lm * data)
        spectrum_coef += np.abs(coef_lm) ** 2
    return spectrum_coef


def stopping_index(
    ns: np.ndarray,
    degree: int = 40,
    thr: float = 1.6e-4,
    kappa_init: np.ndarray | None = None,
) -> int | None:
    """
    Returns stopping index beyond which E^{l_max} / E_total >= thr.

    In rapidly distorted turbulence, the domain distorts rapidly. At some
    point the grid becomes insufficient to capture the distortion. This is
    detected by watching the power spectrum coefficient at a high degree
    spherical harmonic (after normalizing by total PSD): once it exceeds
    thr, the grid is too noisy and the cutoff is enforced.

    Args:
        ns (np.ndarray): Numerical solver state, shape (9, num_nodes, num_time_steps).
        degree (int, optional): Highest spherical harmonic degree tested. Must be
            `2 * (t // 4)` for a spherical design of degree t, the largest even
            degree the design resolves reliably: `ps` integrates the product
            `Y_lm * phi` of degree `l + deg(phi)`, so exactness needs `2l <= t`,
            and odd degrees vanish identically since phi is even in k. thr is
            calibrated at this degree by scripts/earlystopping_thresholds.py.
            Defaults to 40.
        thr (float, optional): Threshold above which case is flagged. Defaults to 1.6e-4.
        kappa_init (np.ndarray | None, optional): Initial wavevectors, shape (3, num_nodes).
            Needed when compound distortions are used and ns[..., 0] does not
            contain the initial wavevectors. Defaults to None, which uses ns[..., 0].

    Returns:
        int | None: Time step index where threshold is first reached, or
            None if threshold is never reached.
    """
    sh = sphericart.SphericalHarmonics(degree)
    if kappa_init is None:
        ns_0 = ns[:, :, 0]
        kx, ky, kz = ns_0[6, ...], ns_0[7, ...], ns_0[8, ...]
    else:
        kx, ky, kz = kappa_init[0, ...], kappa_init[1, ...], kappa_init[2, ...]
    theta = np.arctan2(ky, kx)
    r = np.sqrt(kx**2 + ky**2 + kz**2)
    phi_angle = np.arccos(np.clip(kz / r, -1.0, 1.0))

    x = np.sin(phi_angle) * np.cos(theta)
    y = np.sin(phi_angle) * np.sin(theta)
    z = np.cos(phi_angle)

    # xyz at t0 used for all t; accounts for coordinate transform from
    # deformed to unit sphere. on unit sphere, coordinates come from kappa_0.
    xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])  # (N, 3)

    Y = sh.compute(xyz)  # (N, (degree+1)^2)

    for time_step in tqdm(range(ns.shape[2])):
        ns_t = ns[:, :, time_step]
        phi = ns_t[:6]

        E_phi = float(np.real(psd_total_energy(phi)))

        lmode_phi = 0
        for i in range(phi.shape[0]):
            lmode_phi += ps(phi[i], Y, degree)

        lmode_normalized_phi = lmode_phi / E_phi

        if lmode_normalized_phi >= thr:
            return time_step

    return None


def degree_modes(
    kappa_init: np.ndarray, degree: int, chunk_size: int = 8192
) -> np.ndarray:
    """
    Builds real spherical harmonics of one degree at initial wavevector directions.

    Only tested degree's block of harmonic basis is retained, built in node chunks,
    so peak memory holds chunk_size * (degree + 1)^2 values instead of
    num_nodes * (degree + 1)^2, of which all but 2 * degree + 1 columns per node
    go unused.

    Args:
        kappa_init (np.ndarray): Initial wavevectors, shape (3, num_nodes).
        degree (int): Spherical harmonic degree.
        chunk_size (int, optional): Number of nodes whose harmonics are built at
            once. Defaults to 8192.

    Returns:
        np.ndarray: Harmonics of given degree, shape (num_nodes, 2 * degree + 1).
    """
    kx, ky, kz = kappa_init[0, ...], kappa_init[1, ...], kappa_init[2, ...]
    theta = np.arctan2(ky, kx)
    r = np.sqrt(kx**2 + ky**2 + kz**2)
    phi_angle = np.arccos(np.clip(kz / r, -1.0, 1.0))

    x = np.sin(phi_angle) * np.cos(theta)
    y = np.sin(phi_angle) * np.sin(theta)
    z = np.cos(phi_angle)

    # xyz at t0 used for all t, as in stopping_index
    xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])  # (N, 3)

    # retain only tested degree's orders, columns degree^2 : (degree + 1)^2
    sh = sphericart.SphericalHarmonics(degree)
    first, last = degree * degree, (degree + 1) ** 2
    num_nodes = xyz.shape[0]
    modes = np.empty((num_nodes, last - first))
    for start in range(0, num_nodes, chunk_size):
        stop = start + chunk_size
        modes[start:stop] = sh.compute(xyz[start:stop])[:, first:last]

    return modes


def stopping_index_from_modes(
    phi: np.ndarray, modes: np.ndarray, thr: float = 1.6e-4
) -> int | None:
    """
    Returns stopping index beyond which E^{l_max} / E_total >= thr, given harmonics.

    Orders of tested degree are summed by one matrix product, and every time step
    is evaluated in that same product.

    Args:
        phi (np.ndarray): Packed spectrum, shape (num_time_steps, 6, num_nodes).
        modes (np.ndarray): Harmonics of tested degree from `degree_modes`, shape
            (num_nodes, 2 * degree + 1).
        thr (float, optional): Threshold above which case is flagged. Defaults to 1.6e-4.

    Returns:
        int | None: Time step index where threshold is first reached, or
            None if threshold is never reached.
    """
    assert not np.iscomplexobj(phi), "phi must be real."

    # contiguous so product below stays in BLAS
    phi = np.ascontiguousarray(phi)
    num_time_steps, _, num_nodes = phi.shape
    quadrature = 4.0 * np.pi / num_nodes

    E_phi = quadrature * np.einsum("tin, tin -> t", phi, phi)

    coefficients = quadrature * (phi.reshape(-1, num_nodes) @ modes)
    lmode_phi = (coefficients * coefficients).sum(axis=-1)
    lmode_phi = lmode_phi.reshape(num_time_steps, -1).sum(axis=-1)

    crossed = lmode_phi / E_phi >= thr

    return int(np.argmax(crossed)) if crossed.any() else None


def fast_stopping_index(
    ns: np.ndarray,
    degree: int = 40,
    thr: float = 1.6e-4,
    kappa_init: np.ndarray | None = None,
    chunk_size: int = 8192,
) -> int | None:
    """
    Returns stopping index beyond which E^{l_max} / E_total >= thr.

    Vectorized equivalent of `stopping_index`, returning same index for same
    input, built from `degree_modes` and `stopping_index_from_modes`. Evaluating
    every time step forgoes early exit of `stopping_index`, which is worth it
    since the product costs far less than the loop it replaces.

    Args:
        ns (np.ndarray): Numerical solver state, shape (9, num_nodes, num_time_steps).
        degree (int, optional): Highest spherical harmonic degree tested. Must be
            `2 * (t // 4)` for a spherical design of degree t, as for
            `stopping_index`. Defaults to 40.
        thr (float, optional): Threshold above which case is flagged. Defaults to 1.6e-4.
        kappa_init (np.ndarray | None, optional): Initial wavevectors, shape (3, num_nodes).
            Needed when compound distortions are used and ns[..., 0] does not
            contain the initial wavevectors. Defaults to None, which uses ns[..., 0].
        chunk_size (int, optional): Number of nodes whose harmonics are built at
            once. Defaults to 8192.

    Returns:
        int | None: Time step index where threshold is first reached, or
            None if threshold is never reached.
    """
    if kappa_init is None:
        kappa_init = ns[6:, :, 0]
    modes = degree_modes(kappa_init, degree, chunk_size=chunk_size)

    # (6, N, T) to (T, 6, N)
    return stopping_index_from_modes(ns[:6].transpose(2, 0, 1), modes, thr=thr)


def batched_stopping_index(
    sol: np.ndarray,
    degree: int = 40,
    thr: float = 1.6e-4,
    kappa_init: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns stopping index per case in batch of solver states.

    Harmonic basis is built once for the whole batch, since every case starts
    from the same wavevectors.

    Args:
        sol (np.ndarray): Numerical solver states, shape
            (batch, num_time_steps, 9, num_nodes).
        degree (int, optional): Highest spherical harmonic degree tested. Defaults to 40.
        thr (float, optional): Threshold above which case is flagged. Defaults to 1.6e-4.
        kappa_init (np.ndarray | None, optional): Initial wavevectors, shape
            (3, num_nodes), shared across every case in the batch. Defaults to
            None, which uses sol[:, 0], shared by every case.

    Returns:
        np.ndarray: Stopping index per case, shape (batch,), dtype int. Cases
            that never cross thr are assigned num_time_steps.

    Raises:
        ValueError: If kappa_init is None and cases start from different wavevectors.
    """
    if kappa_init is None:
        kappa_init = sol[0, 0, 6:]
        if not np.all(sol[:, 0, 6:] == kappa_init):
            raise ValueError("Every case must start from the same wavevectors.")
    modes = degree_modes(kappa_init, degree)

    num_time_steps = sol.shape[1]
    es_array = np.empty(sol.shape[0], dtype=int)

    for case in tqdm(range(sol.shape[0]), desc="early stopping"):
        index = stopping_index_from_modes(sol[case, :, :6], modes, thr=thr)
        es_array[case] = num_time_steps if index is None else index

    return es_array


def spectrum_over_time(
    ns: np.ndarray, degree: int = 40, norm: bool = True
) -> tuple[list, list, list]:
    """
    Evaluates spherical-harmonic spectrum of phi, kappa, and kappa_hat over time.

    For each time step, decomposes phi, kappa, and kappa_hat into spherical
    harmonic modes l=0..degree, using the coordinate transform from the
    deformed grid back to the unit sphere at t0 for every time step.

    Args:
        ns (np.ndarray): Numerical solver state, shape (9, num_nodes, num_time_steps).
        degree (int, optional): Highest spherical harmonic degree l_max. Defaults to 40.
        norm (bool, optional): If True, normalizes each l-mode by total energy;
            if False, returns raw l-mode magnitudes. Defaults to True.

    Returns:
        tuple[list, list, list]: spectrum_phi, spectrum_kappa, spectrum_kappa_hat,
            each of length num_time_steps with degree+1 entries per time step.
    """
    spectrum_phi = []
    spectrum_kappa = []
    spectrum_kappa_hat = []
    sh = sphericart.SphericalHarmonics(degree)

    ns_0 = ns[:, :, 0]

    kx, ky, kz = ns_0[6, ...], ns_0[7, ...], ns_0[8, ...]
    theta = np.arctan2(ky, kx)
    r = np.sqrt(kx**2 + ky**2 + kz**2)
    phi_angle = np.arccos(np.clip(kz / r, -1.0, 1.0))

    x = np.sin(phi_angle) * np.cos(theta)
    y = np.sin(phi_angle) * np.sin(theta)
    z = np.cos(phi_angle)

    # xyz at t0 used for all t; accounts for coordinate transform from
    # deformed to unit sphere. on unit sphere, coordinates come from kappa_0.
    xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])  # (N, 3)

    Y = sh.compute(xyz)  # (N, (degree+1)^2)

    for time_step in tqdm(range(ns.shape[2])):
        ns_t = ns[:, :, time_step]

        phi = ns_t[:6]
        kappa = ns_t[6:9]
        kappa_hat = kappa / np.linalg.norm(kappa, axis=0, keepdims=True)
        spectrum_temp_phi = []
        spectrum_temp_kappa = []
        spectrum_temp_kappa_hat = []

        E_phi = psd_total_energy(phi)
        E_kappa = psd_total_energy(kappa)
        E_kappa_hat = psd_total_energy(kappa_hat)

        for l in range(0, degree + 1):
            lmode_phi = 0
            lmode_kappa = 0
            lmode_kappa_hat = 0

            for i in range(phi.shape[0]):
                lmode_phi += ps(phi[i], Y, l)
            for i in range(kappa.shape[0]):
                lmode_kappa += ps(kappa[i], Y, l)
                lmode_kappa_hat += ps(kappa_hat[i], Y, l)

            lmode_normalized_phi = lmode_phi / E_phi
            lmode_normalized_kappa = lmode_kappa / E_kappa
            lmode_normalized_kappa_hat = lmode_kappa_hat / E_kappa_hat

            if norm:
                spectrum_temp_phi.append(lmode_normalized_phi)
                spectrum_temp_kappa.append(lmode_normalized_kappa)
                spectrum_temp_kappa_hat.append(lmode_normalized_kappa_hat)
            else:
                spectrum_temp_phi.append(lmode_phi)
                spectrum_temp_kappa.append(lmode_kappa)
                spectrum_temp_kappa_hat.append(lmode_kappa_hat)

        spectrum_phi.append(spectrum_temp_phi)
        spectrum_kappa.append(spectrum_temp_kappa)
        spectrum_kappa_hat.append(spectrum_temp_kappa_hat)

    return spectrum_phi, spectrum_kappa, spectrum_kappa_hat
