"""Tests get eigs implementation
"""
import numpy as np

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs

from luescher_nd.solver import get_eigs


np.random.seed(42)


def test_get_eigs():
    """Tests get eigs implementation
    """
    n_max = 10000
    n_entries = n_max * 5
    n_eigs = min(n_max // 3, 20)

    rows = np.random.randint(0, high=n_max, size=n_entries)
    cols = np.random.randint(0, high=n_max, size=n_entries)
    vals = np.random.normal(size=n_entries)

    m_sci = lil_matrix((n_max, n_max))

    for i, j, m_ij in zip(rows, cols, vals):
        m_sci[i, j] += m_ij

    m_sci += m_sci.T
    m_sci = m_sci.tocsr()

    e_py = np.sort(eigs(m_sci, k=n_eigs, which="SR", return_eigenvectors=False)).real
    e_cpp = get_eigs(m_sci, n_eigs)

    np.testing.assert_allclose(e_py.real, e_cpp)
