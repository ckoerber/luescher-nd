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


def get_A1g_operator(n1d: int, ndim: int, axis=0) -> sp.csr_matrix:
    """ Implements the projection operator that's schematically | A1g > < A1g |.
    Because A1g is espeically simple, the procedure works in any dimension, specified by ndim.

    If you have a wavefunction |psi> written as a sum of plane waves, applying this operator will give you
        sum_{state j that's in A1g} |j><j|psi>
    If |psi> isn't already in momentum space, who knows what this will do!

    ** Arguments **
        n1d: int
            Number of points in one direction.
        ndim: int
            Number of spatial dimensions.
    """
    ntot = n1d ** ndim
    A1g_mat = sp.dok_matrix((ntot, ntot), dtype=float)

    # One-dimensional momenta, with signs.
    p1d = np.array(list(np.arange(0, (n1d + 1) // 2))
                    + list(-np.arange(n1d // 2, 0, -1))
                    )

    # A list of coordinates:
    momenta = np.transpose([el.flatten() for el in np.meshgrid(*[p1d] * ndim)])

    # Group momenta by norm^2, keeping index
    nsq = dict()
    for i,p in enumerate(momenta):
        n2 = np.dot(p,p)
        if n2 not in nsq:
            nsq[n2] = [[i, p]]
        else:
            nsq[n2]+=[[i, p]]

    # We can't just take nsq by nsq, because some nsq have more than one
    # choice for momenta vectors that don't go into one another under octahedral symmetry.
    # peek=9 # other good choices are 17, 18, 25, ...
    # print(nsq[peek])
    # print(len(nsq[peek]))
    for n2 in nsq:
        # print(f"------------{n2}------------")
        # print(nsq[n2])
        for i, vi in nsq[n2]:
            # Build a list of guys that meet vi under Oh:
            matched = []
            for j, vj in nsq[n2]:
                match = np.sort(np.abs(vi, dtype=int)) == np.sort(np.abs(vj, dtype=int))
                if match.all():
                    matched+=[[j, vj]]

            # With those in hand, we can construct states
            # It's actually here that we specify A1g at all!
            # It's A1g because all the entries get the same entry.
            # A different irrep would require a different pattern.
            # In principle, we can get these by evaluating Ylm spherical harmonics
            # on the momentum vectors---you've got to be careful to norm the guys
            # that live on the boundary of the brillouin zone correctly.
            # How this works I forget; I just do s-wave / A1g, where the answer is
            # just always sum and then norm by the number of states.
            norm = 1/np.sqrt(len(matched))
            for j, vj in matched:
                A1g_mat[i,j] = norm
        # print(handled)
            # We could build a non-square projector P that we could apply to H, as in P @ H @ Pdagger
            # Written that way, it'd take project to A1g states, dropping all non-A1g from the Hamiltonian.
            # That'd make the Hamiltonian much much much smaller, and therefore (presumably) easier to diagonalize.
            # However, it requires sparse-dense-sparse mat-mat-mat multiply.
    return A1g_mat.tocsr()

def get_A1g_projector(n1d: int, ndim: int):
    # The A1g_operator should have eigenvalues 1 on A1g states and 0 on non-A1g states.
    # Therefore, to project the other guys out, we need 1-A1g.
    # That will be multiplied by something large, giving non-A1g states a big boost in energy.
    a1g = get_A1g_operator(n1d, ndim)
    one = sp.eye(n1d ** ndim)
    # return one - a1g
    return a1g

def get_A1g_reducer(n1d: int, ndim: int, axis=0) -> sp.csr_matrix:
    """ Implements the projection operator that's schematically | A1g > < A1g |.
    Because A1g is espeically simple, the procedure works in any dimension, specified by ndim.

    If you have a wavefunction |psi> written as a sum of plane waves, applying this operator will give you
        sum_{state j that's in A1g} |j><j|psi>
    If |psi> isn't already in momentum space, who knows what this will do!

    ** Arguments **
        n1d: int
            Number of points in one direction.
        ndim: int
            Number of spatial dimensions.
    """
    ntot = n1d ** ndim
    A1g_mat = []

    # One-dimensional momenta, with signs.
    p1d = np.array(list(np.arange(0, (n1d + 1) // 2))
                    + list(-np.arange(n1d // 2, 0, -1))
                    )

    # A list of coordinates:
    momenta = np.transpose([el.flatten() for el in np.meshgrid(*[p1d] * ndim)])

    # Group momenta by norm^2, keeping index
    nsq = dict()
    for i,p in enumerate(momenta):
        n2 = np.dot(p,p)
        if n2 not in nsq:
            nsq[n2] = [[i, p]]
        else:
            nsq[n2]+=[[i, p]]

    # We can't just take nsq by nsq, because some nsq have more than one
    # choice for momenta vectors that don't go into one another under octahedral symmetry.
    # peek=9 # other good choices are 17, 18, 25, ...
    # print(nsq[peek])
    # print(len(nsq[peek]))
    for n2 in nsq:
        # print(f"------------{n2}------------")
        # print(nsq[n2])
        handled = []
        for i, vi in nsq[n2]:
            if i in handled:
                continue
            # Build a list of guys that meet vi under Oh:
            matched = []
            for j, vj in nsq[n2]:
                if j in handled:
                    continue
                match = np.sort(np.abs(vi, dtype=int)) == np.sort(np.abs(vj, dtype=int))
                if match.all():
                    matched+=[[j, vj]]

            # With those in hand, we can construct states
            # It's actually here that we specify A1g at all!
            # It's A1g because all the entries get the same entry.
            # A different irrep would require a different pattern.
            # In principle, we can get these by evaluating Ylm spherical harmonics
            # on the momentum vectors---you've got to be careful to norm the guys
            # that live on the boundary of the brillouin zone correctly.
            # How this works I forget; I just do s-wave / A1g, where the answer is
            # just always sum and then norm by the number of states.
            norm = 1/np.sqrt(len(matched))
            vector = np.zeros(n1d**ndim)
            for j, vj in matched:
                vector[j] = norm
                handled += [j]

            A1g_mat += [vector]
        # print(handled)
            # We could build a non-square projector P that we could apply to H, as in P @ H @ Pdagger
            # Written that way, it'd take project to A1g states, dropping all non-A1g from the Hamiltonian.
            # That'd make the Hamiltonian much much much smaller, and therefore (presumably) easier to diagonalize.
            # However, it requires sparse-dense-sparse mat-mat-mat multiply.
    return np.array(A1g_mat)
