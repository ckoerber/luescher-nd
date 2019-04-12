"""Operators
"""
import numpy as np

import scipy.sparse as sp


def get_parity_projector(n1d: int, ndim: int, cutoff: float = 100):
    """
    """
    ntot = n1d ** ndim
    parity_mat = sp.dok_matrix([ntot, ntot], dtype=int)

    exponent = n1d ** np.arange(ndim)
    for nxi in np.transpose(
        [el.flatten() for el in np.meshgrid(*[np.arange(n1d)] * ndim)]
    ):
        nr = np.dot(nxi, n1d ** exponent)
        nrp = np.dot((-nxi) % n1d, n1d ** exponent)
        parity_mat[nrp, nr] = 1

    return sp.coo_tocsr(sp.eye(ntot) + cutoff * (sp.eye(ntot) - parity_mat))
