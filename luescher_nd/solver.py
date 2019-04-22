"""Solver routines for easy eigenvalue interface
"""
from typing import Optional
from typing import Dict

from dataclasses import dataclass
from dataclasses import field

import logging
import itertools

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lina

try:
    from solvers.src import cupy_sp
    import cupy as cp  # pylint: disable=E0401, W0611
except ModuleNotFoundError:
    cupy_sp = None  # pylint: disable=C0103

from luescher_nd.utilities import get_logger

from luescher_nd.hamiltonians.kinetic import get_kinetic_hamiltonian
from luescher_nd.hamiltonians.contact import get_full_hamiltonian
from luescher_nd.hamiltonians.contact import MomentumContactHamiltonian

LOGGER = get_logger(logging.INFO)


@dataclass
class Solver:
    """Solver for fast eigenvalue access depending on contact interaction strength.
    """

    n1d_max: int
    lattice_spacing: float = 1.0
    particle_mass: float = 4.758
    ndim_max: int = 3
    derivative_shifts: Optional[Dict[int, float]] = field(default=None, repr=False)
    kinetic_hamiltonian: sp.lil_matrix = field(init=False, repr=False)
    cuda: bool = field(default=False, repr=False)
    mom_space: bool = False
    nstep: Optional[int] = None

    def __post_init__(self):
        self.derivative_shifts = self.derivative_shifts or {-1: 1.0, 0: -2.0, 1: 1.0}
        LOGGER.debug("Allocating kinetic hamiltonian")
        if not self.mom_space:
            self.kinetic_hamiltonian = get_kinetic_hamiltonian(
                self.n1d_max,
                self.lattice_spacing,
                self.particle_mass,
                self.ndim_max,
                self.derivative_shifts,
                self.cuda,
            )
        else:
            self.kinetic_hamiltonian = MomentumContactHamiltonian(
                n1d=self.n1d_max,
                epsilon=self.lattice_spacing,
                m=self.particle_mass,
                ndim=self.ndim_max,
                c0=0.0,
                nstep=self.nstep,
            )

    def get_ground_state(self, contact_strength: float, **kwargs) -> float:
        """Returns smallest algebraic eigenvalue in of Hamiltonian in `[fm]`

        Arguments
        ---------
            contact_strength: float
                The strength of the contact interaction in respective units.
                This depends on the dimension of the problem, e.g., [fm]**(-1 - ndim).

            kwargs:
                Additional keyword arguments passed to `eigsh` solver.
        """

        if not self.mom_space:
            H = get_full_hamiltonian(
                self.kinetic_hamiltonian,
                contact_strength,
                self.ndim_max,
                self.lattice_spacing,
                self.cuda,
            )
        else:
            H = MomentumContactHamiltonian(
                n1d=self.n1d_max,
                epsilon=self.lattice_spacing,
                m=self.particle_mass,
                ndim=self.ndim_max,
                c0=contact_strength,
                nstep=self.nstep,
            ).mat
        LOGGER.debug("Computing ground state")
        if cupy_sp and self.cuda:
            out = cupy_sp.lanczos_cp(H, n_eigs=1, **kwargs)[0]
        else:
            out = lina.eigsh(H, k=1, which="SA", **kwargs)[0][0]

        return out

    def get_energies(
        self, contact_strength: float, n_energies: int = 1, **kwargs
    ) -> float:
        """Returns smallest algebraic eigenvalues in of Hamiltonian in `[fm]`

        Arguments
        ---------
            contact_strength: float
                The strength of the contact interaction in respective units.
                This depends on the dimension of the problem, e.g., [fm]**(-1 - ndim).

            n_energies: int
                Number of energies.

            kwargs:
                Additional keyword arguments passed to `eigsh` solver.
        """
        if not self.mom_space:
            H = get_full_hamiltonian(
                self.kinetic_hamiltonian,
                contact_strength,
                self.ndim_max,
                self.lattice_spacing,
                self.cuda,
            )
        else:
            H = MomentumContactHamiltonian(
                n1d=self.n1d_max,
                epsilon=self.lattice_spacing,
                m=self.particle_mass,
                ndim=self.ndim_max,
                c0=contact_strength,
                nstep=self.nstep,
            ).mat
        LOGGER.debug("Computing eigenvalues")
        if cupy_sp and self.cuda:
            if "max_iter" not in kwargs:
                kwargs["max_iter"] = n_energies + 10
            out = cupy_sp.lanczos_cp(H, n_eigs=n_energies, **kwargs)
        else:
            out = lina.eigsh(
                H, k=n_energies, which="SA", return_eigenvectors=False, **kwargs
            )
        return out


def get_approx_psi0(
    particle_energy: float,
    n1d_max: int,
    ndim_max: int,
    lattice_spacing: float = 1.0,
    particle_mass: float = 4.758,
) -> np.array:
    r"""Computes educated guess for bound state solution.

    Returns the normalized asymptotic wave function
    $$
        \psi(r) = A \frac{ \exp{ - \gamma r } }{r}
    $$
    with $\gamma = \sqrt{ - 2 m_p E  }$ and $A$ being chosen such that
    $$
        a_L^n_{\rm dim} \sum_{r} \psi^2(r) = 1 \, .
    $$

    Arguments
    ---------
        particle_energy: float
            The energy of the particle used to approximate the ground state.

        n1d_max: int
            Number of spatial lattice nodes in one dimension.

        n1d_max: int
            Number of spatial dimensions.

        lattice_spacing: float
            Distance between two spatial lattice nodes (homogenious grid).

        particle_mass:
            The mass of an individual particle used to approximate the ground state.
    """
    r = []
    for nxi in itertools.product(*[range(n1d_max)] * ndim_max):
        r.append(np.sqrt(np.sum(np.array(nxi) ** 2)) * lattice_spacing)
    r = np.array(r)

    kappa = np.sqrt(-2 * particle_mass * particle_energy)

    psi0 = np.exp(-kappa * r) / (r + 1.0e-7)
    psi0 /= np.sqrt(psi0 @ psi0 * lattice_spacing ** ndim_max)

    return psi0
