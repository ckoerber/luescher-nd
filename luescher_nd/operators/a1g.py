import numpy as np
import luescher_nd.lattice as lattice
import scipy.sparse as sp

import itertools

oneByDegeneracy = {
    1: 1.000_000_000_000_000_00,
    2: 0.500_000_000_000_000_00,
    3: 0.333_333_333_333_333_33,
    4: 0.250_000_000_000_000_00,
    6: 0.166_666_666_666_666_67,
    8: 0.125_000_000_000_000_00,
    12: 8.333_333_333_333_333_33e-2,
    24: 4.166_666_666_666_666_67e-2,
    48: 2.083_333_333_333_333_33e-2,
}

oneBySqrtDegeneracy = {
    1: 1.000_000_000_000_000_0,
    2: 0.707_106_781_186_547_52,
    3: 0.577_350_269_189_625_76,
    4: 0.500_000_000_000_000_00,
    6: 0.408_248_290_463_863_02,
    8: 0.353_553_390_593_273_76,
    12: 0.288_675_134_594_812_88,
    24: 0.204_124_145_231_931_51,
    48: 0.144_337_567_297_406_44,
}


def partners(n1d, ndim, vector):
    sign = [+1, -1]
    momenta = lattice.momenta(n1d, ndim)
    prtners = set()
    for permutation in itertools.permutations(vector):
        for signs in itertools.product(sign, repeat=ndim):
            vector = tuple([p * s for p, s in zip(permutation, signs)])
            if vector in momenta:
                prtners.add(tuple([p * s for p, s in zip(permutation, signs)]))

    return prtners


def partner_count(n1d, ndim, vector):
    return len(partners(n1d, ndim, vector))


def nsq_degeneracy(n1d, ndim, nsq=None):
    if nsq is None:
        primitives = lattice.all_nsq_primitives(n1d, ndim)
        return {nsq: len(primitives[nsq]) for nsq in primitives}
    elif type(nsq) is list:
        return [nsq_degeneracy(n1d, ndim, n2) for n2 in nsq]
    else:
        return len(nsq_primitives(n1d, ndim, nsq))


def projector(n1d: int, ndim: int) -> sp.csr_matrix:  # pylint: disable=R0914
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

    momenta = lattice.momenta(n1d, ndim)
    lookup = lattice.momentum_lookup(n1d, ndim)

    primitives = lattice.all_nsq_primitives(n1d, ndim)

    for nsq in primitives:
        for primitive in primitives[nsq]:
            prtners = partners(n1d, ndim, primitive)
            print(f"nsq={nsq}, {prtners}")
            indices = [lookup[p] for p in prtners]
            norm = oneByDegeneracy[len(indices)]
            for i, j in itertools.product(indices, repeat=2):
                A1g_mat[i, j] = norm

    return A1g_mat.tocsr()


def complement(n1d: int, ndim: int) -> sp.csr_matrix:
    """Computes one minus `get_projector_to_a1g`

    ** Arguments **
        n1d: int
            Number of points in one direction.
        ndim: int
            Number of spatial dimensions.
    """
    # The A1g_operator should have eigenvalues 1 on A1g states and 0 on non-A1g states.
    # Therefore, to project the other guys out, we need 1-A1g.
    # That will be multiplied by something large, giving non-A1g states a big boost in energy.
    a1g = projector(n1d, ndim)
    one = sp.eye(n1d ** ndim)
    return one - a1g


def reducer(n1d: int, ndim: int) -> np.ndarray:  # pylint: disable=R0914
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
    A1g_mat = []

    sites = lattice.sites(n1d, ndim)
    lookup = lattice.momentum_lookup(n1d, ndim)
    primitives = lattice.all_nsq_primitives(n1d, ndim)

    for n2 in primitives:
        for p in primitives[n2]:
            vector = np.zeros(sites)
            prtners = partners(n1d, ndim, p)
            norm = oneBySqrtDegeneracy[len(prtners)]
            for partner in prtners:
                vector[lookup[partner]] = norm

            A1g_mat += [vector]

    return np.array(A1g_mat)
