"""Operators
"""
import numpy as np

import scipy.sparse as sp


def get_parity_operator(n1d: int, ndim: int) -> sp.csr_matrix:
    """Implements the parity operator ``P |psi+/-> = +/- |psi+/->`` for relative coords.

    The operator sends the coordinate ``|p> -> |-p>`` (or ``|r>``) modulo the box
    boundary.

    **Arguments**
        n1d: int
            Number of lattice sites in one dimension.

        ndim: int
            Number of dimensions.
    """
    ntot = n1d ** ndim
    parity_mat = sp.dok_matrix((ntot, ntot), dtype=int)

    exponent = n1d ** np.arange(ndim)

    for nxi in np.transpose(
        [el.flatten() for el in np.meshgrid(*[np.arange(n1d)] * ndim)]
    ):
        nr = np.dot(nxi, exponent)
        nrp = np.dot((-nxi) % n1d, exponent)
        parity_mat[nrp, nr] = 1

    return parity_mat.tocsr()


def get_parity_projector(n1d: int, ndim: int, positive: bool = True):
    """Operator which shifts parity eigenvalues of the wrong parity to large numbers.

    For example if ``positive=True``: ``P+ |psi+> = |psi+>`` and
    ``P+ |psi-> = 0 |psi->``, where ``P |psi+/-> = +/- |psi+/->`` are partiy
    eigenstates of opposite parity.

    **Arguments**
        n1d: int
            Number of lattice sites in one dimension.

        ndim: int
            Number of dimensions.

        positive: bool = True
            Parity component which does not change under operator application.
    """
    p = get_parity_operator(n1d, ndim)
    one = sp.eye(n1d ** ndim)
    return (one + p if positive else one - p) / 2
