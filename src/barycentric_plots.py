import matplotlib.pyplot as plt
import numpy as np

def anisotropy(
    tensor_ij: np.ndarray, reference_ij: np.ndarray | None = None
) -> np.ndarray:
    """
    Computes the anisotropy of a second-order tensor.

    Normalizes by the trace of reference_ij, so D_ij can be normalized by the
    trace of R_ij rather than its own. Trailing axes are arbitrary, covering both
    (3, 3, batch, time) tensors and (3, 3, num_samples) flattened ones.

    Args:
        tensor_ij (np.ndarray): Second-order tensor, shape (3, 3, ...).
        reference_ij (np.ndarray | None, optional): Tensor supplying the
            normalizing trace, same shape as tensor_ij. Defaults to None, which
            uses tensor_ij itself.

    Returns:
        np.ndarray: Anisotropy tensor, same shape as tensor_ij.
    """
    if reference_ij is None:
        reference_ij = tensor_ij

    trace = np.einsum("ii... -> ...", reference_ij)
    identity = np.eye(3).reshape((3, 3) + (1,) * (tensor_ij.ndim - 2))
    return tensor_ij / trace - identity / 3


def barycentric_map_point(
    b_ij,
    color='tab:blue',
    plot=True,
    markersize=80,    
    edgecolor='black',
    linewidth=0.5,
    alpha=0.8,
):
    '''
    b_ij: Reynolds stress anisotropy.
        shape (3, 3) 
    Plots the anisotropy invariants on the 
    Barycentric map. 
    '''

    x1, x2, x3 = 1, 0, 1/2 
    y1, y2, y3 = 0, 0, 3**0.5/2
    
    evals, evecs = np.linalg.eigh(b_ij)

    l3, l2, l1 = np.sort(evals)

    C1 = l1 - l2 
    C2 = 2*(l2 - l3)
    C3 = 3*l3 + 1 

    x = C1*x1 + C2*x2 + C3*x3 
    y = C1*y1 + C2*y2 + C3*y3 

    if plot:
        plt.scatter(
            x, y,
            s=markersize,
            c=color,
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
            zorder=3,
        )
    return [x, y]


def barycentric_map_outline(label = 'C'):
    '''
    The outlines of the triangle 
    that forms the barycentric map. 
    '''
    
    plt.plot([1,0], [0, 0], color = 'k', linewidth = 2.2)
    plt.plot([1,1/2], [0, 3**0.5/2], color = 'k', linewidth = 2.2)
    plt.plot([1/2,0], [3**0.5/2, 0], color = 'k', linewidth = 2.2)
    
    plt.text(-0.1, 0.05, f'${{2{label}}}$', fontsize = 20)
    plt.text(0.95, 0.10, f'${{1{label}}}$', fontsize = 20)
    plt.text(0.55, 0.85, f'${{3{label}}}$', fontsize = 20)
    return 1 