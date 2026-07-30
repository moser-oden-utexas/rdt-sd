"""Computes the structure tensors from the velocity spectrum tensor."""
# R, D, Q, Qs, M, L, J, M6 


import numpy as np 

def anisotropy_tensor(rij):
    '''
    R_ij --> b_ij. 
    
    '''
    tke_rii = rij[0,:] + rij[1,:] + rij[2,:]
    bij = rij/tke_rii
    bij[:3,:]-= 1/3
    return bij 

def evaluate_reynolds_stress_tensor(phi, grid = "spherical_designs"):
    '''
    Velocity Spectrum Tensor --> Reynolds Stress. 
    
    Inputs: 
        Phi: Velocity Spectrum Tensor 
            Size: [6, num_nodes, num_time_steps]
        grid: The grid on which the wavenumbers are 
            Initialized. 
            Options: ["lebedev", "spherical_design", "uniform"]
    Outputs: 
        r_ij: RS. 
            Size: [6, num_time_steps]
    '''
    num_nodes = phi.shape[1]

    if grid != "spherical_designs": 
        
        E2 = " has not been implemented, try spherical designs :)"
        raise Exception("Post processing for " + grid + E2)
    else: 

        C_pre = 0.15 
        uniform_weight = C_pre*4*np.pi/num_nodes
        rij = uniform_weight*np.sum(phi, axis = 1)

    return rij 

def evaluate_reynolds_stress_anisotropy(phi, grid = "spherical_designs"):
    '''
    Velocity Spectrum Tensor --> Reynolds Stress Anisotropy. 
    
    Inputs: 
        Phi: Velocity Spectrum Tensor 
            Size: [6, num_nodes, num_time_steps]
        grid: The grid on which the wavenumbers are 
                    Initialized. 
                    Options: ["lebedev", "spherical_design", "uniform"]
    Outputs: 
        b_ij: RS Anisotropies. 
            Size: [6, num_time_steps]
    '''
        
    rij = evaluate_reynolds_stress_tensor(phi, grid)
    return anisotropy_tensor(rij) 

EPS = np.array([[[0, 0, 0], [0, 0, 1], [0, -1, 0]],
                [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]]) 

def get_M_star_from_M(M):
    '''
    M* - fully symmetric part of M. 

    Inputs: 
        M: shape - [*, 3, 3, 3, 3]
    Outputs: 
        M*: shape - [*, 3, 3, 3, 3]
    Note: 
        It assumes M is symmetric in 
        ij and pq. M_ijpq. 
    '''
    Ms = np.zeros(M.shape)
    Ms += np.einsum('tijpq -> tijpq', M)
    Ms += np.einsum('tipqj -> tijpq', M)
    Ms += np.einsum('tiqjp -> tijpq', M)
    Ms += np.einsum('tpjiq -> tijpq', M)
    Ms += np.einsum('tqjip -> tijpq', M)
    Ms += np.einsum('tpqij -> tijpq', M)

    return 1/6*Ms 

def Q_star_from_Q(Qijk):
    '''
    Q --> Q* 
    
    Q: Q_ijk. 
    Q*: The Symmetric part of the Q tensor.
    Q*: Stropholysis tensor. 
    
    Inputs: 
        Q: Q_ijk Tensor. 
            Shape: [num_cases, 3,3,3]
    Outputs;
        Q*: Symm of Q. 
            Shape: [num_cases, 3,3,3]
            
    Generally, Q* is used in modelling and not Q. 
    '''
    
    Qsijk = np.zeros(Qijk.shape)

    Qsijk += 1/6*(np.einsum('tijk -> tijk', Qijk))
    Qsijk += 1/6*(np.einsum('tikj -> tijk', Qijk))
    Qsijk += 1/6*(np.einsum('tjik -> tijk', Qijk))
    Qsijk += 1/6*(np.einsum('tjki -> tijk', Qijk))
    Qsijk += 1/6*(np.einsum('tkij -> tijk', Qijk))
    Qsijk += 1/6*(np.einsum('tkji -> tijk', Qijk))

    return Qsijk 

def rapid_pressure_strain_rate(M, G): 
    '''
    RPSR:
        T_ij = 2.G_{nm}.(M_{imnj} + M_{jmni})

    Inputs:
        M: Mijpq 
            Shape-[*, 3, 3, 3, 3]
        G: Mean velocity gradient 
            Shape-[*, 3, 3]
    Outputs: 
        Tij - RPSR 
            Shape - [*, 3, 3]

    Reference: 2.8.6 in Kassinos. 
    '''
    
    Tij = 2*np.einsum('timnj, tnm -> tij', M, G)
    # Tij += 2*np.einsum('tjmni, tnm -> tij', M, G)
    Tij += 2*np.einsum('tjmni, tnm -> tij', M, G)

    return Tij 

def evaluate_M_es(phi,
                  k_arr,
                  grid = 'spherical_designs'): 
    '''
    Uses es : einstein summation 
    to evaluate the M tensor 
    using k arr, phi arr. 

    Inputs: 
        phi: (6, num nodes, time steps)
        k_arr: (3, num nodes, time steps)
        grid: dictates how integration is 
            carried out. 
            
    Outputs: 
        M
        (3, 3, 3, 3, num_time_steps)

    Verification: 
        es version has been verified to be 
        the same as the non es version: 

        np.allclose(M_es, M, atol = 1e-16)
        hold True. 
    '''
    k_mag = np.einsum('int, int-> nt', k_arr, k_arr)**0.5 

    # normalized k array 
    k_arr_norm = np.einsum('int, nt -> int', k_arr, 1/k_mag)

    num_nodes, num_time_steps = phi.shape[1:]
    
    phi_arr = np.zeros([3, 3, num_nodes, num_time_steps])

    phi_arr[0,0,:] = phi[0,:]
    phi_arr[1,1,:] = phi[1,:]
    phi_arr[2,2,:] = phi[2,:]
    
    phi_arr[0,1,:] = phi[3,:]
    phi_arr[0,2,:] = phi[4,:]
    phi_arr[1,2,:] = phi[5,:]
    
    phi_arr[1,0,:] = phi[3,:]
    phi_arr[2,0,:] = phi[4,:]
    phi_arr[2,1,:] = phi[5,:]
    
    # t - num time steps 
    # p, q - ki indices 
    # n - num nodes (summed)
    
    M_es = np.einsum('pnt, qnt, ijnt -> ijpqt',
                     k_arr_norm,
                     k_arr_norm,
                     phi_arr)

    if grid == 'spherical_designs': 

        # this uniform weight converts 
        # the values in the shell onto 
        # all the other infinitesimal 
        # shells and gives the statistics 
        # for the while spehere. 
        
        C_pre = 0.15 
        uniform_weight = C_pre*4*np.pi/num_nodes
        return M_es*uniform_weight 
    else: 
        raise Exception("Only S.D. exists, change grid.")
    
def evaluate_L_es(phi,
                  k_arr,
                  grid = 'spherical_designs'): 
    '''
    Uses es : einstein summation 
    to evaluate the L tensor 
    using k arr, phi arr. 

    Inputs: 
        phi: (6, num nodes, time steps)
        k_arr: (3, num nodes, time steps)
        grid: dictates how integration is 
            carried out. 
            
    Outputs: 
        L
        (3, 3, 3, 3, num_time_steps)

    Verification: 
        es version has been verified to be 
        the same as the non es version: 

        np.allclose(M_es, M, atol = 1e-16)
        hold True. 
    '''
    k_mag = np.einsum('int, int-> nt', k_arr, k_arr)**0.5 

    # normalized k array 
    k_arr_norm = np.einsum('int, nt -> int', k_arr, 1/k_mag)

    num_nodes, num_time_steps = phi.shape[1:]
    
    phi_arr = np.zeros([3, 3, num_nodes, num_time_steps])

    phi_arr[0,0,:] = phi[0,:]
    phi_arr[1,1,:] = phi[1,:]
    phi_arr[2,2,:] = phi[2,:]
    
    phi_arr[0,1,:] = phi[3,:]
    phi_arr[0,2,:] = phi[4,:]
    phi_arr[1,2,:] = phi[5,:]
    
    phi_arr[1,0,:] = phi[3,:]
    phi_arr[2,0,:] = phi[4,:]
    phi_arr[2,1,:] = phi[5,:]
    
    # t - num time steps 
    # p, q - ki indices 
    # n - num nodes (summed)
    
    M_es = np.einsum('pnt, qnt, int, jnt, kknt -> ijpqt',
                     k_arr_norm,
                     k_arr_norm,
                     k_arr_norm,
                     k_arr_norm,
                     phi_arr)

    if grid == 'spherical_designs': 

        # this uniform weight converts 
        # the values in the shell onto 
        # all the other infinitesimal 
        # shells and gives the statistics 
        # for the while spehere. 
        
        C_pre = 0.15 
        uniform_weight = C_pre*4*np.pi/num_nodes
        return M_es*uniform_weight 
    else: 
        raise Exception("Only S.D. exists, change grid.")


def evaluate_J_5th_order_es(M_6th):
    '''
    Evaluates the 5th order J 
    tensors using the 6th order 
    M tensor. 

    J_ijrpq = \eps_its M_sjtrpq 

    Verification: 
        J_ijrpp = Q_ijr 
        holds True. 
    '''
    
    return np.einsum('its, sjtrpqn -> ijrpqn', EPS, M_6th)

def evaluate_M_6th_order_es(phi,
                  k_arr,
                  grid = 'spherical_designs'): 
    '''
    Uses es : einstein summation 
    to evaluate the M tensor 
    using k arr, phi arr. 

    Inputs: 
        phi: (6, num nodes, time steps)
        k_arr: (3, num nodes, time steps)
        grid: dictates how integration is 
            carried out. 
            
    Outputs: 
        M
        (3, 3, 3, 3, 3, 3, num_time_steps)

    Verification: 
        M_ijpqrs can be converted to M_ijpq using 
        the following contraction. 

        M_ijpprs = M_ijrs 
        M_ijpqrr = M_ijpq 

        This has been verified. 
    '''
    k_mag = np.einsum('int, int-> nt', k_arr, k_arr)**0.5 

    # normalized k array 
    k_arr_norm = np.einsum('int, nt -> int', k_arr, 1/k_mag)

    num_nodes, num_time_steps = phi.shape[1:]
    
    phi_arr = np.zeros([3, 3, num_nodes, num_time_steps])

    phi_arr[0,0,:] = phi[0,:]
    phi_arr[1,1,:] = phi[1,:]
    phi_arr[2,2,:] = phi[2,:]
    
    phi_arr[0,1,:] = phi[3,:]
    phi_arr[0,2,:] = phi[4,:]
    phi_arr[1,2,:] = phi[5,:]
    
    phi_arr[1,0,:] = phi[3,:]
    phi_arr[2,0,:] = phi[4,:]
    phi_arr[2,1,:] = phi[5,:]

    # d- num time steps 
    # t, r, p, q - ki indices 
    # n - num nodes (summed)
    
    M_es = np.einsum('tnd, rnd, pnd, qnd, sjnd -> sjtrpqd',
                     k_arr_norm,
                     k_arr_norm,
                     k_arr_norm,
                     k_arr_norm,
                     phi_arr)

    if grid == 'spherical_designs': 

        # this uniform weight converts 
        # the values in the shell onto 
        # all the other infinitesimal 
        # shells and gives the statistics 
        # for the while spehere. 
        
        C_pre = 0.15 
        uniform_weight = C_pre*4*np.pi/num_nodes
        return M_es*uniform_weight 
    else: 
        raise Exception("Only S.D. exists, change grid.")



rij_from_Mijpq = lambda M: np.einsum('tijpp -> tij', M)
dij_from_Mijpq = lambda M: np.einsum('tiipq -> tpq', M)
Qijk_from_Mijpq = lambda M, lc: np.einsum('ipq, tjqpk -> tijk', EPS, M)
htke_from_Mijpq = lambda M: np.einsum('tiipp -> t', M) 