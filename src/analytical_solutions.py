import numpy as np 
from src.spherical_designs import init_wavenumbers_spherical_designs 


def analytical_solution_pure_shear(sd_t_degree, 
                                      t_max,
                                        time_steps,
                                        rot_matrix = None, 
                                        alpha = 1): 
    '''
    Using the solution from 
    Hunt et al. (2022), the analytical 
    solution to phi, karr are evaluated 
    for time [0, t] over time_steps. 

    For pure shear, the velocity gradient is
    
    G = [[0, 0, alpha],
         [0, 0, 0],
         [0, 0, 0]]

    Inputs: 
        sd_t_degree - spherical design degree for wavenumbers
        t_max - final time 
        time_steps - number of time steps
        alpha - shear magnitude  
        rotation matrix - if provided, 
            calculates the solution in the rotated frame. 
    Outputs:
        phi_array_t - (time_steps, num_nodes, 3, 3) 
                      array of phi at each time step
        k_array_t - (time_steps, num_nodes, 3) 
                    array of karr at each time step 
    
    Notes:
        The RDT set up we used has the principal strain direction 
        fixed at x. So, to match numerical solution 
        with analytical solution, the analytical solution 
        is evaluated in an appropriate rotated frame. 
        ^rot mean rotated quantity below: 
            -> phi_anl^rot(k^rot) = R^T phi_anl(k) R
            -> R^T phi_anl(k) R = R^T phi_anl(R k^rot) R 
    '''

    def phi_arr_t(phi_arr0, t): 
        '''
        Computed the analytical solution 
        for pure shear rdt 
        '''
        phi_arr = np.zeros_like(phi_arr0)

        phi_arr[:, 0, 0] = phi_arr0[:, 0, 0] 
        phi_arr[:, 0, 0] += eps_1(t)*phi_arr0[:, 0, 2] 
        phi_arr[:, 0, 0] += eps_1(t)*phi_arr0[:, 2, 0]
        phi_arr[:, 0, 0] += eps_1(t)**2*phi_arr0[:, 2, 2]

        phi_arr[:, 0, 1] = phi_arr0[:, 0, 1]
        phi_arr[:, 0, 1] += eps_2(t)*phi_arr0[:, 0, 2]
        phi_arr[:, 0, 1] += eps_1(t)*phi_arr0[:, 2, 1]
        phi_arr[:, 0, 1] += eps_1(t)*eps_2(t)*phi_arr0[:, 2, 2]

        phi_arr[:, 1, 0] = phi_arr[:, 0, 1].copy() 

        phi_arr[:, 0, 2] = phi_arr0[:, 0, 2]*k0**2/k(t)**2
        phi_arr[:, 0, 2] += eps_1(t)*phi_arr0[:, 2, 2]*k0**2/k(t)**2

        phi_arr[:, 2, 0] = phi_arr[:, 0, 2].copy()

        phi_arr[:, 1, 1] = phi_arr0[:, 1, 1]
        phi_arr[:, 1, 1] += eps_2(t)*phi_arr0[:, 1, 2]
        phi_arr[:, 1, 1] += eps_2(t)*phi_arr0[:, 2, 1]
        phi_arr[:, 1, 1] += eps_2(t)**2*phi_arr0[:, 2, 2]

        phi_arr[:, 1, 2] = phi_arr0[:, 1, 2]*k0**2/k(t)**2
        phi_arr[:, 1, 2] += eps_2(t)*phi_arr0[:, 2, 2]*k0**2/k(t)**2

        phi_arr[:, 2, 1] = phi_arr[:, 1, 2].copy()

        phi_arr[:, 2, 2] = phi_arr0[:, 2, 2]*k0**4/k(t)**4

        return phi_arr  

    _karr, _ = init_wavenumbers_spherical_designs(t = sd_t_degree)
    karr = np.einsum('ik, tk -> ti', rot_matrix, _karr) if rot_matrix is not None else _karr 

    k1 = karr[:, 0]
    k2 = karr[:, 1]
    k30 = karr[:, 2]

    k0 = np.sqrt(k1**2 + k2**2 + k30**2)

    k3 = lambda t: k30 - k1*alpha*t 
    k = lambda t: np.sqrt(k1**2 + k2**2 + k3(t)**2)
    C1 = lambda t:  k1*(k30*k(t)**2 - k3(t)*k0**2)/(1e-18 + (k1**2 + k2**2))/(1e-18 + k(t)**2 ) 

    C2 = lambda t: k2*k0**2/(1e-18+(k1**2 + k2**2)**1.5)*np.arctan2((k30 - k3(t))*np.sqrt(k1**2 + k2**2), k1**2 + k2**2 + k30*k3(t))

    eps_1 = lambda t: C1(t) - k2/(1e-18+k1)*C2(t) 
    eps_2 = lambda t: k2/(1e-18+k1)*C1(t) + C2(t) 

    K_mat = np.einsum('ti,tj->tij', karr, karr)/np.einsum('ti,ti->t', karr, karr)[:, None, None]

    delta = np.eye(3)[None, :, :]

    phi_arr0 = delta - K_mat

    phi_array_updates = [] 
    for t in np.linspace(0, t_max, time_steps): 
        phi_array_updates.append(phi_arr_t(phi_arr0, t))

    def karr_t(karr0, t): 
        karr = karr0.copy() 
        karr[:, 2] = karr0[:, 2] - karr0[:, 0]*alpha*t 
        return karr

    karr_updates = [] 
    for t in np.linspace(0, t_max, time_steps): 
        karr_updates.append(karr_t(karr, t))

    _phi_array_t = np.array(phi_array_updates)
    k_array_t = np.array(karr_updates)

    if rot_matrix is not None:
        phi_array_t = np.einsum('ij, tnjk, kl -> tnil', rot_matrix.T, _phi_array_t, rot_matrix)
        k_array_t   = np.einsum('ij, tnj -> tni', rot_matrix.T, k_array_t)
    else:
        phi_array_t = _phi_array_t

    return phi_array_t, k_array_t 



def phi_analytical_strain(sd_t_degree,
                          t_max,
                           num_time_steps,
                           s11 = 1, s22 = -0.5, s33 = -0.5): 
    '''
    Analytical solution for pure strain. 
    Solution referred from MJ Lee (1986, 1989).

    The mean velocity gradient for the pure
    strain case is given by: 

    G = [[s11, 0, 0],
         [0, s22, 0],
         [0, 0, s33]]. 
    
    Inputs: 
    1. sd_t_degree - spherical design degree for wavenumbers
    2. t_max - final time
    3. num_time_steps - number of time steps to evaluate the solution at

    Outputs:
    phi_array, k_array 
    (num_time_steps, 6, num_nodes), (num_time_steps, 3, num_nodes)
    
    Notes: 
        1. Using s11, s22, s33, any general pure strain case 
        can be evaluated.        
        2. The output shape of phi which this function is different from 
    the output shape of the pure shear case. Here only the unique 
    6 components are returned, but the in pure shear a (3,3) tensor 
    is returned for phi. 
    '''
    
    assert s11 + s22 + s33 == 0, "For incompressible flow, the sum of the strain rates should be 0." 

    k_arr0, num_nodes = init_wavenumbers_spherical_designs(t = sd_t_degree)

    e1 = lambda t: np.exp(s11*t)
    e2 = lambda t: np.exp(s22*t)
    e3 = lambda t: np.exp(s33*t)

    def update_karr_strain(karr0, t):
        k1, k2, k3 = karr0[:, 0], karr0[:, 1], karr0[:, 2]
        k1_t = k1/e1(t)
        k2_t = k2/e2(t)
        k3_t = k3/e3(t)
        karr = np.vstack([k1_t, k2_t, k3_t])
        return karr 

    def phi_arr_strain(k_arr, t): 

        k1, k2, k3 = k_arr[:, 0], k_arr[:, 1], k_arr[:, 2]

        pre_factor = 1 
        # E(k)/4pi k^2
        # in current solver is initialized to 1, as the numerical solver 
        # solves on a spherical shell 

        phi_arr = np.zeros([6, num_nodes])

        chi_sq = k1**2/e1(t)**2 + k2**2/e2(t)**2 + k3**2/e3(t)**2

        phi_arr[0, :] = pre_factor/chi_sq**2*(e3(t)**2/e2(t)**2*k2**2*(k1**2 + k2**2) + e2(t)**2/e3(t)**2*k3**2*(k1**2 + k3**2) + 2*k2**2*k3**2)
        phi_arr[1, :] = pre_factor/chi_sq**2*(e1(t)**2/e3(t)**2*k3**2*(k2**2 + k3**2) + e3(t)**2/e1(t)**2*k1**2*(k2**2 + k1**2) + 2*k3**2*k1**2) 
        phi_arr[2, :] = pre_factor/chi_sq**2*(e2(t)**2/e1(t)**2*k1**2*(k3**2 + k1**2) + e1(t)**2/e2(t)**2*k2**2*(k3**2 + k2**2) + 2*k1**2*k2**2)

        k_sq = k1**2 + k2**2 + k3**2 

        # Phi_12, Phi_13, Phi_23  
        phi_arr[3, :] = -pre_factor/chi_sq**2*(k1*k2/(e1(t)*e2(t)))*( (k_sq - k3**2)/(e1(t)**2 * e2(t)**2) + (k3**2/e3(t)**2)*(1/e1(t)**2 + 1/e2(t)**2 - 1/e3(t)**2) )
        phi_arr[4, :] = -pre_factor/chi_sq**2*(k1*k3/(e1(t)*e3(t)))*( (k_sq - k2**2)/(e1(t)**2 * e3(t)**2) + (k2**2/e2(t)**2)*(1/e1(t)**2 + 1/e3(t)**2 - 1/e2(t)**2) )
        phi_arr[5, :] = -pre_factor/chi_sq**2*(k2*k3/(e2(t)*e3(t)))*( (k_sq - k1**2)/(e2(t)**2 * e3(t)**2) + (k1**2/e1(t)**2)*(1/e2(t)**2 + 1/e3(t)**2 - 1/e1(t)**2) )

        return phi_arr 

    phi_array_updates = []
    for t in np.linspace(0, t_max, num_time_steps): 
        phi_array_updates.append(phi_arr_strain(k_arr0, t))

    karr_updates = []
    for t in np.linspace(0, t_max, num_time_steps): 
        karr_updates.append(update_karr_strain(k_arr0, t))

    karr_t = np.array(karr_updates)
    phi_array_t = np.array(phi_array_updates) 
    
    return phi_array_t, karr_t 

