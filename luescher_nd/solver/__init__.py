"""Import wrapper for C++ implementation sparse solver
"""

import numpy as np
import scipy.sparse as sp

from luescher_nd.solver.solver import get_eigs_py


def get_eigs(sparse_mat: sp.spmatrix, n_eigs: int):
    """Compute eigenvalues of sparse matrix

    Computes smallest real eigenvalues.

    ** Arguments **
        sparse_mat: SparseMatrix
            A scipy sparse matrix type

        n_eigs:
            Number of eigenvalues to compute.
    """
    m = sparse_mat.todok()
    rows, cols = np.array(list(m.keys())).T
    return get_eigs_py(n_eigs, sparse_mat.shape[0], rows, cols, m.values())
