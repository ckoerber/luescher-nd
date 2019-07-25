

import numpy as np
import luescher_nd.lattice as lattice
import scipy.sparse as sp


def operator(n1d: int, ndim: int) -> sp.csr_matrix:
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


def projector(n1d: int, ndim: int, positive: bool = True):
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
    p = operator(n1d, ndim)
    one = sp.eye(n1d ** ndim)
    return (one + p if positive else one - p) / 2


def get_90_rotation_operator(n1d: int, axis=0) -> sp.csr_matrix:
    """Implements the z-rotation operator ``Rz(90)  |psi(p)> = |psi(Rz(90)p)>``

    Rotation is implemented in relative coords.

    This operator is explicitly implemented for 3D.

    The operator sends the coordinate
    ``|Rz(90)p> -> |Rz(90)(p_x, p_y, p_z)> = |Rz(90)(p_y, -p_x, p_z)>``
    modulo boundaries.

    **Arguments**
        n1d: int
            Number of lattice sites in one dimension.

        ndim: int
            Number of dimensions.
    """
    ndim = 3
    ntot = n1d ** ndim
    parity_mat = sp.dok_matrix((ntot, ntot), dtype=int)

    exponent = n1d ** np.arange(ndim)

    rotation = np.zeros([3, 3], dtype=int)
    a1, a2 = set([0, 1, 2]).remove(axis)
    rotation[a1, a2] = 1
    rotation[a2, a1] = -1
    rotation[axis, axis] = 1

    for nxi in np.transpose(
        [el.flatten() for el in np.meshgrid(*[np.arange(n1d)] * ndim)]
    ):
        nr = np.dot(nxi, exponent)
        nrp = np.dot((rotation @ nxi) % n1d, exponent)
        parity_mat[nrp, nr] = 1

    return parity_mat.tocsr()
