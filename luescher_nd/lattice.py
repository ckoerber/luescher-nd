
import numpy as np

def n(n1d):
    return np.array(
        list(np.arange(0, (n1d + 1) // 2)) + list(-np.arange(n1d // 2, 0, -1))
    )

def momenta(n1d, ndim):
    return np.transpose([el.flatten() for el in np.meshgrid(*[n(n1d)] * ndim)])
