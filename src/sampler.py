from pathlib import Path

from SALib.sample.sobol import sample
import numpy as np

def sampling_parameters(num_cases: int, use_coriolis: bool, seed: int = 42) -> np.ndarray:
    """
    Samples RDT parameters via Sobol sequence.

    Always draws the 4 mean-velocity-gradient parameters (s1..s4). Draws the 3
    Coriolis parameters (c1..c3) too when `use_coriolis` is True, so every
    sampled Sobol dimension maps onto a parameter that is actually used, and
    the generated tensors stay as unique as the underlying Sobol sequence.

    Args:
        num_cases (int): Number of cases to sample.
        use_coriolis (bool): Whether to sample the Coriolis parameters.
        seed (int, optional): Sobol sampler seed. Defaults to 42.

    Returns:
        np.ndarray: Sampled parameters, shape (num_cases, 4) when use_coriolis
            is False, or (num_cases, 7) when True.
    """
    names = ["s1", "s2", "s3", "s4"]
    bounds = [[-1, 1], [-1, -0.5], [0, 2 * np.pi], [-1, 1]]
    if use_coriolis:
        names += ["c1", "c2", "c3"]
        bounds += [[0, 10], [0, 2 * np.pi], [-1, 1]]

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}

    # sobol sampler gives ((2 + D + D*(D-1)/2) * N cases
    # where D is number of parameters
    N = int(np.ceil(np.log2(num_cases)))

    param_values = sample(problem, 2**N, seed=seed, calc_second_order=True)

    return param_values[:num_cases, :]


def generate_mean_velocity_gradients(params):
    """
    Generate mean velocity-gradient tensors from sampled strain/rotation parameters.

    Each row of `params` defines one velocity-gradient tensor:

        params[i] = [alpha, s22, azimuth, cos_polar]

    where:
        alpha:
            Controls the strain/rotation blend.

            The symmetric strain part is

                S = alpha * diag([1, s22, -1 - s22])

            so `alpha = 0` gives pure rotation and `abs(alpha) = 1`
            gives pure strain.

        s22:
            Second principal strain component. The third component is chosen
            as `s33 = -1 - s22`, so the strain tensor is trace-free.

        azimuth:
            Azimuthal angle of the rotation-rate direction in the e1-e2 plane,
            in radians.

        cos_polar:
            Cosine of the polar angle measured from the e3 axis. This is clipped
            to [-1, 1] before applying arccos for numerical safety.

    The rotation-rate direction is

        w = [
            sin(polar) * cos(azimuth),
            sin(polar) * sin(azimuth),
            cos(polar),
        ]

    and the antisymmetric part is constructed as

        W_ij = -epsilon_ijk w_k

    scaled by `(1 - abs(alpha))`.

    Args:
        params: Array with shape (num_cases, 4), containing
            [alpha, s22, azimuth, cos_polar] for each case.

    Returns:
        np.ndarray:
            Array of mean velocity gradients with shape (num_cases, 3, 3).
    """
    params = np.asarray(params, dtype=float)

    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError(f"Expected params with shape (num_cases, 4), got {params.shape}")

    param_values = params.copy()
    param_values[:, -1] = np.arccos(np.clip(param_values[:, -1], -1.0, 1.0))

    G_array = []

    for alpha, s2, azimuth, polar in param_values:
        w1 = np.sin(polar) * np.cos(azimuth)
        w2 = np.sin(polar) * np.sin(azimuth)
        w3 = np.cos(polar)

        S = alpha * np.diag([1.0, s2, -1.0 - s2])

        W = (1.0 - abs(alpha)) * np.array([
            [0.0,  w3, -w2],
            [-w3, 0.0,  w1],
            [w2, -w1, 0.0],
        ])

        G_array.append(S + W)

    return np.array(G_array)


def generate_coriolis_terms(params):
    """
    Generate Coriolis rotation vectors from sampled spherical parameters.

    Each row of `params` defines one Coriolis vector:

        params[i] = [omega_mag, azimuth, cos_polar]

    where:
        omega_mag:
            Magnitude of the Coriolis/rotation vector.

        azimuth:
            Azimuthal angle in the e1-e2 plane, in radians.

        cos_polar:
            Cosine of the polar angle measured from the e3 axis. This is clipped
            to [-1, 1] before applying arccos for numerical safety.

    The resulting unit direction is

        w = [
            sin(polar) * cos(azimuth),
            sin(polar) * sin(azimuth),
            cos(polar),
        ]

    and the returned vector is

        omega = omega_mag * w

    Args:
        params: Array with shape (num_cases, 3), containing
            [omega_mag, azimuth, cos_polar] for each case.

    Returns:
        np.ndarray:
            Coriolis vectors with shape (num_cases, 3).
    """
    params = np.asarray(params, dtype=float)

    if params.ndim != 2 or params.shape[1] != 3:
        raise ValueError(f"Expected params with shape (num_cases, 3), got {params.shape}")

    param_values = params.copy()
    param_values[:, -1] = np.arccos(np.clip(param_values[:, -1], -1.0, 1.0))

    omega_array = []

    for omega_mag, azimuth, polar in param_values:
        w1 = np.sin(polar) * np.cos(azimuth)
        w2 = np.sin(polar) * np.sin(azimuth)
        w3 = np.cos(polar)

        omega_array.append(omega_mag * np.array([w1, w2, w3]))

    return np.array(omega_array)


def sample_case_parameters(
    num_cases: int, use_coriolis: bool, seed: int = 42
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Samples mean velocity gradients and, when enabled, Coriolis terms.

    Args:
        num_cases (int): Number of cases to sample.
        use_coriolis (bool): Whether to sample and generate Coriolis terms.
        seed (int, optional): Sobol sampler seed. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Mean velocity gradients, shape
            (num_cases, 3, 3), and Coriolis terms, shape (num_cases, 3), or
            None when use_coriolis is False.
    """
    sampling_params = sampling_parameters(num_cases, use_coriolis, seed=seed)
    mean_velocity_gradients = generate_mean_velocity_gradients(sampling_params[:, :4])
    coriolis_terms = (
        generate_coriolis_terms(sampling_params[:, 4:]) if use_coriolis else None
    )
    return mean_velocity_gradients, coriolis_terms


def _load_case_parameters(
    grad_u_location: str | Path,
    coriolis_location: str | Path | None,
    use_coriolis: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Loads mean velocity gradients and Coriolis terms from .npy files.

    Args:
        grad_u_location (str | Path): Path to .npy file holding mean velocity
            gradients, shape (num_cases, 3, 3).
        coriolis_location (str | Path | None): Path to .npy file holding Coriolis
            terms, shape (num_cases, 3), or None when use_coriolis is False.
        use_coriolis (bool): Whether to load and apply Coriolis terms.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Mean velocity gradients, shape
            (num_cases, 3, 3), and Coriolis terms, shape (num_cases, 3), or
            None when use_coriolis is False.

    Raises:
        ValueError: If either array has an unexpected shape, or if the two files
            hold different numbers of cases.
    """
    mean_velocity_gradients = np.asarray(np.load(grad_u_location), dtype=float)
    if mean_velocity_gradients.ndim != 3 or mean_velocity_gradients.shape[1:] != (3, 3):
        raise ValueError(
            f"grad_u_location must hold an array of shape (num_cases, 3, 3), got "
            f"{mean_velocity_gradients.shape}."
        )

    if not use_coriolis:
        return mean_velocity_gradients, None

    coriolis_terms = np.asarray(np.load(coriolis_location), dtype=float)
    if coriolis_terms.ndim != 2 or coriolis_terms.shape[1] != 3:
        raise ValueError(
            f"coriolis_location must hold an array of shape (num_cases, 3), got "
            f"{coriolis_terms.shape}."
        )

    if mean_velocity_gradients.shape[0] != coriolis_terms.shape[0]:
        raise ValueError(
            f"grad_u_location holds {mean_velocity_gradients.shape[0]} cases, "
            f"coriolis_location holds {coriolis_terms.shape[0]}."
        )

    return mean_velocity_gradients, coriolis_terms


def resolve_case_parameters(
    num_cases: int,
    use_coriolis: bool,
    seed: int = 42,
    grad_u_location: str | Path | None = None,
    coriolis_location: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Resolves the case pool, either by Sobol sampling or from explicit .npy files.

    Sobol sampling is the default. When `grad_u_location` is given, cases are
    loaded from disk instead and `num_cases` and `seed` are ignored, so the pool
    size is the number of entries in the file.

    Args:
        num_cases (int): Number of cases to sample. Ignored in file mode.
        use_coriolis (bool): Whether to use Coriolis terms.
        seed (int, optional): Sobol sampler seed. Defaults to 42. Ignored in
            file mode.
        grad_u_location (str | Path | None, optional): Path to .npy file holding
            mean velocity gradients, shape (num_cases, 3, 3). Defaults to None,
            which samples instead.
        coriolis_location (str | Path | None, optional): Path to .npy file holding
            Coriolis terms, shape (num_cases, 3). Defaults to None. Required
            alongside grad_u_location when use_coriolis is True.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Mean velocity gradients, shape
            (num_cases, 3, 3), and Coriolis terms, shape (num_cases, 3), or
            None when use_coriolis is False.

    Raises:
        ValueError: If the two locations are combined inconsistently with each
            other or with use_coriolis, or if the loaded arrays are malformed.
    """
    if grad_u_location is None and coriolis_location is not None:
        raise ValueError("grad_u_location is required when coriolis_location is given.")

    if not use_coriolis and coriolis_location is not None:
        raise ValueError("coriolis_location requires use_coriolis to be true.")

    if grad_u_location is None:
        return sample_case_parameters(num_cases, use_coriolis, seed=seed)

    if use_coriolis and coriolis_location is None:
        raise ValueError("coriolis_location is required when use_coriolis is true.")

    return _load_case_parameters(grad_u_location, coriolis_location, use_coriolis)


if __name__ == "__main__":
    num_cases = 100
    sampling_parameters(num_cases, use_coriolis=True, seed=42)


    
