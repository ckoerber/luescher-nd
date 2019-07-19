"""Test for projection operators
"""
from itertools import product

from unittest import TestCase

import numpy as np

from luescher_nd.hamiltonians.contact import MomentumContactHamiltonian

from luescher_nd.operators import get_parity_operator
from luescher_nd.operators import get_projector_to_a1g


def commutator(op: "LinearOperator", sparse: "SparseMatrix") -> "Matrix":
    """Computes the commutator between an operator and a sparse matrix.
    """
    return op @ sparse.toarray() - sparse @ (op @ np.eye(op.shape[0]))


def matrix_norm(mat: "Matrix") -> float:
    """Returns the trace of `sqrt(mat.T @ mat)`
    """
    return np.sqrt(np.diag(mat.T @ mat).sum())


class ProjectorTest(TestCase):
    """Compares momentum space computations to coordinate space results
    """

    places = 10
    longMessage = True

    ranges = {
        "n1d": [3, 4],
        "nstep": [1, 2, None],
        "epsilon": [1.0, 0.1],
        "mass": [1.0, 0.5],
        "ndim": [1, 3],
        "contact_strength": [0.0, -1.0],
    }

    def setUp(self):
        """Prepares Hamiltonians
        """
        self.parameters = []
        for vals in product(*self.ranges.values()):
            self.parameters.append(
                {key: val for key, val in zip(self.ranges.keys(), vals)}
            )

        self.hamiltonians = self._hamiltonian_generator()

    def _hamiltonian_generator(self):
        """
        """
        for pars in self.parameters:
            yield MomentumContactHamiltonian(**pars)

    def test_01_dispersion_parity_commutator(self):
        """Checks if matrix norm of commutator of H0 and parity operator is zero
        """
        for h in self.hamiltonians:
            with self.subTest(h=h):
                p = get_parity_operator(h.n1d, h.ndim)
                p2 = np.diag(h._disp_over_m)  # pylint: disable=W0212
                commu = p @ p2 - p2 @ p
                self.assertAlmostEqual(
                    matrix_norm(commu),
                    0,
                    places=self.places,
                    msg=f"Failed test for hamiltonian: {h}",
                )

    def test_02_dispersion_a1g_commutator(self):
        """Checks if matrix norm of commutator of H0 and a1g projector is zero
        """
        for h in self.hamiltonians:
            with self.subTest(h=h):
                p = get_projector_to_a1g(h.n1d, h.ndim)
                p2 = np.diag(h._disp_over_m)  # pylint: disable=W0212
                commu = p @ p2 - p2 @ p
                self.assertAlmostEqual(
                    matrix_norm(commu),
                    0,
                    places=self.places,
                    msg=f"Failed test for hamiltonian: {h}",
                )

    def test_03_contact_parity_commutator(self):
        """Checks if matrix norm of commutator of contact H and parity operator is zero
        """
        for h in self.hamiltonians:
            with self.subTest(h=h):
                p = get_parity_operator(h.n1d, h.ndim)
                commu = commutator(h.op, p)
                self.assertAlmostEqual(
                    matrix_norm(commu),
                    0,
                    places=self.places,
                    msg=f"Failed test for hamiltonian: {h}",
                )

    def test_04_contact_a1g_commutator(self):
        """Checks if matrix norm of commutator of contact H and a1g projector is zero
        """
        for h in self.hamiltonians:
            with self.subTest(h=h):
                p = get_projector_to_a1g(h.n1d, h.ndim)
                commu = commutator(h.op, p)
                self.assertAlmostEqual(
                    matrix_norm(commu),
                    0,
                    places=self.places,
                    msg=f"Failed test for hamiltonian: {h}",
                )
