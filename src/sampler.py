from SALib.sample.sobol import sample
import numpy as np 

def sampling_parameters(num_cases, second_order=True, seed = 42):
    '''
    Sample 7 parameters for the RDT simulation using a Sobol sequence.
        -4 for the mean velocity gradient.  
        -3 for the coriolis terms.
    Inputs: 
        - num_cases: Number of samples to generate
    Outputs:
        - params: numpy array of shape (num_cases, 7) containing the sampled parameters
    ''' 


    problem = {
        'num_vars': 7,
        'names': ['s1', 's2', 's3', 's4', 'c1', 'c2', 'c3'],
        'bounds': [[-1, 1],
                [-1, -0.5],
                [0, 2*np.pi], 
                [-1, 1], 
                [0, 10], 
                [0, 2*np.pi], 
                [-1, 1]]
    }

    # sobol sampler gives ((2 + D + D*(D-1)/2) * N cases 
    # where D is number of parameters

    N = int(np.ceil(np.log2(num_cases))) 

    param_values = sample(problem,
                        2**N,
                        seed = seed, 
                        calc_second_order=second_order)
    
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

if __name__ == "__main__":
    num_cases = 100 
    sampling_parameters(num_cases, second_order=True, seed=42)


    
