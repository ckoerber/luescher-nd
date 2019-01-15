"""Import wrapper for C++ implementation sparse solver
"""

from luescher_nd.solver.solver import get_eigs_py


def get_eigs(sparse_mat, n_eigs):
    """
    """
    return get_eigs_py(
        n_eigs,
        sparse_mat.shape[0],
        sparse_mat.indices,
        sparse_mat.indptr,
        sparse_mat.data,
    )
