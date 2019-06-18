"""Test for comparing eigenvalues computed in different spaces.
"""
from itertools import product

from unittest import TestCase
from unittest import skip

from numpy import sort
from numpy.testing import assert_allclose
from scipy.sparse.linalg import eigsh

from luescher_nd.utilities import get_laplace_coefficients
from luescher_nd.hamiltonians.contact import MomentumContactHamiltonian


@skip("Currently disabled because solver changes")
class MomentumSpaceTest(TestCase):
    """Compares momentum space computations to coordinate space results
    """

    ranges = {
        "n1d": [3, 4],
        "nstep": [1, 2],
        "epsilon": [1.0, 0.1],
        "m": [1.0, 0.5],
        "ndim": [1, 3],
        "c0": [0.0, -1.0],
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
            solver_coord = Solver(
                pars["n1d"],
                lattice_spacing=pars["epsilon"],
                particle_mass=pars["m"],
                ndim_max=pars["ndim"],
                derivative_shifts=get_laplace_coefficients(pars["nstep"]),
            )
            h_mom = MomentumContactHamiltonian(**pars)
            yield solver_coord, h_mom

    def test_01_free_eigenvalues(self):
        """Compares free eigenvalues of coord space Hamiltonians with mom space H0.
        """
        for solver_coord, h_mom in self.hamiltonians:
            if abs(h_mom.c0) > 1.0e-12:
                continue
            with self.subTest(solver_coord=solver_coord, h_mom=h_mom):
                neigs = h_mom.n1d ** h_mom.ndim - 2
                mom_eigs = eigsh(h_mom.mat, neigs, which="SA", return_eigenvectors=False)
                coord_eigs = solver_coord.get_energies(h_mom.c0, neigs)
                assert_allclose(sort(coord_eigs), sort(mom_eigs), atol=1.0e-10)

    def test_02_eigenvalues(self):
        """Compares eigenvalues of coord space Hamiltonians with mom space equivalent.
        """
        for solver_coord, h_mom in self.hamiltonians:
            with self.subTest(solver_coord=solver_coord, h_mom=h_mom):
                neigs = h_mom.n1d ** h_mom.ndim - 2
                mom_eigs = eigsh(h_mom.mat, neigs, which="SA", return_eigenvectors=False)
                coord_eigs = solver_coord.get_energies(h_mom.c0, neigs)
                assert_allclose(sort(coord_eigs), sort(mom_eigs), atol=1.0e-10)
