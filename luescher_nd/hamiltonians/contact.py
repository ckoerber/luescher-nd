"""Implementation of contact Hamiltonians
"""
from dataclasses import dataclass

import logging

import numpy as np
from scipy import sparse as sp

try:
    from solvers.src import cupy_sp
    import cupy as cp  # pylint: disable=E0401, W0611
except ModuleNotFoundError:
    cupy_sp = None  # pylint: disable=C0103

from luescher_nd.utilities import get_logger

from luescher_nd.hamiltonians.kinetic import MomentumKineticHamiltonian

LOGGER = get_logger(logging.INFO)


def get_full_hamiltonian(
    kinetic_hamiltonian: sp.csr_matrix,
    contact_strength: float,
    ndim_max: int = 3,
    lattice_spacing: float = 1.0,
    cuda: bool = False,
) -> sp.csr_matrix:
    r"""Copies kinetic Hamiltonian and adds contact strength from the (0, 0) component.

    Computes `H = H0 + V`, where `V = contact_strength * delta(r)` and `nr` is ther
    relative distance in the two particle system.

    Note that the discrete delta function becomes
    $$
        \frac{\delta_{n_r, 0}}{a_L^n_dim} \, .
    $$
    This is implemented in this routine.

    Arguments
    ---------
        kinetic_hamiltonian: sparse matrix
            The kinetic two-body lattice Hamiltonian.

        contact_strength: float
            The strength of the contact interaction in respective units.
            This depends on the dimension of the problem, e.g., [fm]**(-1 - ndim).
    """
    LOGGER.debug("Allocating full hamiltonian")
    contact_interaction = sp.lil_matrix(kinetic_hamiltonian.shape, dtype=float)
    contact_interaction[(0, 0)] = contact_strength / lattice_spacing ** ndim_max
    if cupy_sp and cuda:
        contact_interaction = cupy_sp.scipy2cupy(  # pylint: disable=E1101
            contact_interaction.tocsr()
        )

    return (contact_interaction + kinetic_hamiltonian).tocsr()


@dataclass(frozen=True)
class MomentumContactHamiltonian(MomentumKineticHamiltonian):
    """Contact interaction Hamiltonian in momentum space
    """

    c0: float = -1.0

    def apply(self, vec):
        """Applies hamiltonian to vector
        """
        return self._disp_over_m * vec + self.c0 * np.sum(vec) / self.L ** self.ndim
