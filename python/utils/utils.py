import numpy as np

def simplex_projection(v):
    r"""Compute the projection of `v` on the simplex.

    Args:
        v (np.ndarray): Input vector.
    """

    # Some black magic for simplex projection (found by solving KKT conditions on the projection minimization problem)
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = 1 - np.cumsum(u)
    ind = np.arange(n) + 1
    cond = u + cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v + theta, 0)
    return w