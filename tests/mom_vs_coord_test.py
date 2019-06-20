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
from luescher_nd.hamiltonians.kinetic import get_kinetic_hamiltonian
from luescher_nd.hamiltonians.contact import get_full_hamiltonian


class MomentumSpaceTest(TestCase):
    """Compares momentum space computations to coordinate space results
    """

    ranges = {
        "n1d": [3, 4],
        "nstep": [1, 2],
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
            h_coord_kin = get_kinetic_hamiltonian(
                pars["n1d"],
                lattice_spacing=pars["epsilon"],
                particle_mass=pars["mass"],
                ndim_max=pars["ndim"],
                derivative_shifts=get_laplace_coefficients(pars["nstep"]),
            )
            h_coord = get_full_hamiltonian(
                h_coord_kin,
                contact_strength=pars["contact_strength"],
                ndim_max=pars["ndim"],
                lattice_spacing=pars["epsilon"],
            )
            h_mom = MomentumContactHamiltonian(**pars)
            yield h_coord, h_mom

    def test_01_free_eigenvalues(self):
        """Compares free eigenvalues of coord space Hamiltonians with mom space H0.
        """
        for h_coord, h_mom in self.hamiltonians:
            with self.subTest(solver_coord=h_coord, h_mom=h_mom):
                neigs = h_mom.n1d ** h_mom.ndim - 2
                mom_eigs = eigsh(h_mom.op, neigs, which="SA", return_eigenvectors=False)
                coord_eigs = eigsh(h_coord, neigs, which="SA", return_eigenvectors=False)
                assert_allclose(sort(coord_eigs), sort(mom_eigs), atol=1.0e-10)

    def test_02_eigenvalues(self):
        """Compares eigenvalues of coord space Hamiltonians with mom space equivalent.
        """
        for h_coord, h_mom in self.hamiltonians:
            with self.subTest(solver_coord=h_coord, h_mom=h_mom):
                neigs = h_mom.n1d ** h_mom.ndim - 2
                mom_eigs = eigsh(h_mom.op, neigs, which="SA", return_eigenvectors=False)
                coord_eigs = eigsh(h_coord, neigs, which="SA", return_eigenvectors=False)
                assert_allclose(sort(coord_eigs), sort(mom_eigs), atol=1.0e-10)
