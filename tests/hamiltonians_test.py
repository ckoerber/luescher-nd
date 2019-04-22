# pylint: disable=W0212
"""Unittests for luescher_nd.utilites
"""
from itertools import product
from unittest import TestCase

import numpy as np

from luescher_nd.utilities import get_laplace_coefficients
from luescher_nd.hamiltonians.kinetic import get_kinetic_hamiltonian


class TestLaplaceCoefficients(TestCase):
    """Tests coefficients for nstep derivative
    """

    n_step_max: int = 4

    @staticmethod
    def _get_laplace_coefficients(n_step: int):
        """Implementation fo finite step laplace derivative coefficients.

        See also https://en.wikipedia.org/wiki/Finite_difference_coefficient.

        Arguments
        ---------
            n_step: int
                Step range of derivative. Must be larger than 0.
        """
        out = {}

        if n_step < 1:
            raise ValueError("'n_step' must be larger than zero.")

        elif n_step == 1:
            out[0] = -2
            out[1] = out[-1] = 1

        elif n_step == 2:
            out[0] = -5 / 2
            out[1] = out[-1] = 4 / 3
            out[2] = out[-2] = -1 / 12

        elif n_step == 3:
            out[0] = -49 / 18
            out[1] = out[-1] = 3 / 2
            out[2] = out[-2] = -3 / 20
            out[3] = out[-3] = 1 / 90

        elif n_step == 4:
            out[0] = -205 / 72
            out[1] = out[-1] = 8 / 5
            out[2] = out[-2] = -1 / 5
            out[3] = out[-3] = 8 / 315
            out[4] = out[-4] = -1 / 560

        else:
            raise NotImplementedError(
                "'n_step' not implemented for values larger than 6."
            )

        return out

    def test_coefficients(self):
        """Tests values for `luescher_nd.utilities.get_laplace_coefficients` coeffiencts.

        Compares results against Wikipedia counterparts.
        """
        for n in range(1, self.n_step_max + 1):
            with self.subTest(n_step_max=n):
                computed = get_laplace_coefficients(n)
                for nstep, coeff in self._get_laplace_coefficients(n).items():
                    self.assertAlmostEqual(coeff, computed[nstep], places=12)


class TestKineticHamiltonian(TestCase):
    """Test case for kinetic Hamiltonian
    """

    def setUp(self):
        """Initializes kinetic Hamiltonian
        """

        self.n1d = 4
        self.ndim = 2
        self.shifts = {-1: 1.0, 0: -2.0, 1: 1.0}
        self.lattice_spacing = 1.24
        self.mass = 0.356

        self.kinetic_hamiltonian = get_kinetic_hamiltonian(
            n1d_max=self.n1d,
            ndim_max=self.ndim,
            derivative_shifts=self.shifts,
            lattice_spacing=self.lattice_spacing,
            particle_mass=self.mass,
        ).toarray()

    def test_hermiticity(self):
        r"""Test if $H_0^\dagger = H_0$
        """
        np.testing.assert_equal(
            self.kinetic_hamiltonian, self.kinetic_hamiltonian.T.conj()
        )

    def test_eigenvalues(self):
        r"""Test if eigenvalues aggree with discrete finite volume dispersion relation.

        The eigenvlues of the discrete kinetic Hamiltonian are given by
        $$
            E(\vec p) = - \frac{1}{M_N \epsilon^2}\left[
                \sum_{n_d = 1}^{n_{dim}}
                \sum_{n_s = - n_{step}}^{n_{step}}
                c_{n_s} \cos( p_{n_d}  n_{step} \epsilon )
            \right] \, ,
        $$

        where $p_L$ are the lattice momenta defined by
        $$
            p_{n_d}(\vec n, \vec \phi)
            =
            \frac{2 \pi n_{n_d}}{L} + \frac{\phi_{n_d} }{L}
            \, .
        $$

        The vector $\vec n$ runs over all allowed lattice grid points.
        """
        energies = []
        l1d = self.n1d * self.lattice_spacing

        for n in product(*[range(self.n1d)] * self.ndim):
            En = 0
            for ni in n:
                pi = 2 * np.pi * ni / l1d

                for nstep, coeff in self.shifts.items():
                    En -= coeff * np.cos(pi * nstep * self.lattice_spacing)

            energies.append(En)

        energies = np.array(energies) / self.mass / self.lattice_spacing ** 2

        eigs, _ = np.linalg.eigh(self.kinetic_hamiltonian)

        # Sort to enable comparism
        energies.sort()
        eigs.sort()

        np.testing.assert_array_almost_equal(energies, eigs)
