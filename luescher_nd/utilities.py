# pylint: disable=E1101
"""Module for computing expectation values of kinetic hamiltonian in n dimensions.
"""
import itertools
from dataclasses import dataclass
from dataclasses import field

import logging

from typing import Optional
from typing import Dict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lina

try:
    from solvers.src import cupy_sp
    import cupy as cp
except ModuleNotFoundError:
    cupy_sp = None  # pylint: disable=C0103


def get_logger(level):
    """Creates a logger a given level
    """
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(name)s|%(asctime)s] %(message)s")
    ch.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.addHandler(ch)
    return logger


LOGGER = get_logger(logging.INFO)


def get_kinetic_hamiltonian(  # pylint: disable=R0914
    n1d_max: int,
    lattice_spacing: float = 1.0,
    particle_mass: float = 4.758,
    ndim_max: int = 3,
    derivative_shifts: Optional[Dict[int, float]] = None,
    cuda: bool = False,
) -> sp.csr_matrix:
    r"""Computes the kinetic Hamiltonian for a relative two-body system in a 3D box.

    The kinetic Hamiltonian is defined by
        `H0 = - Laplace / 2 /(particle_mass/2)`
    Where Laplace is the lattice laplace operator with twisted boundary conditions.

    The function uses scipy sparse matrices in column format.

    Arguments
    ---------
        n1d_max: int
            Spatial dimension of lattice in any direction.

        lattice_spacing: float
            The lattice spacing in fermi.

        particle_mass: float
            The mass of the particles to be described in inverse fermi.

        ndim_max: float
            The number of spatial dimensions.

        derivative_shifts: Dict[int, float] or None
            The implementation of the numerical laplace operator.
            E.g., for a one step derivative
                `d2f/dx2(n) = sum(
                    c_i * f(n + s_i) * exp(1j * twist_angle/n1d_max * s_i)
                )/lattice_spacing**2`
            where s_i and c_i are the keys and values of derivative_shifts.
            If None, the shifts are defaulted to {-1: 1.0, 0: -2.0, 1: 1.0}.

    Long description
    ----------------
    In the one-dimensional case, a simple discrete 1-step derivative for periodic
    boundary conditions is given by
    $$
        \frac{d^2 f(x = n \epsilon)}{dx^2}
        \rightarrow \frac{1}{\epsilon^2}
        \left[
            f(n\epsilon + \epsilon)
            - 2 f(n\epsilon)
            + f(n\epsilon - \epsilon)
        \right] \, ,
    $$
    where $\epsilon$ is the lattice spacing.

    For improved discrete derivative implementations (n-steps) see also
    [Finite difference coefficients]
    (https://en.wikipedia.org/wiki/Finite_difference_coefficient).

    The kinetic Hamiltonian in relative coordinates is computed by dividing the above
    expression by two times the reduced mass of the subsystem &mdash;
    in the case of equal mass particles, half the mass of individual particles.

    The below implementation of the kinetic Hamiltonian is general in the number of
    dimensions and the implementation of the derivative.
    This is achieved by creating a meta index for coordinates:
    ```python
    nr = n1 + n2 * n1d_max ** 1 + n3 * n1d_max ** 2 + ...
    ```
    where `ni` describe the index in each respective dimension.

    Taking the derivative in a respective direction adjusts the `ni` components.
    Because the grid is finite with boundary conditions, all `ni` are identified with
    their `n1d_max` modulus.

    For example, when shifitng the y-coordinate in a three dimensional space by one
    lattice spacing becomes
    ```
    nr_yshift = nx + ((ny + 1) % n1d_max) * n1d_max ** 1 + nz * n1d_max ** 2
    ```

    The below kinetic Hamiltonian iterates over all dimensions and shifts to create
    a `n_max x n_max` matrix with `n_max = n1d_max ** ndim_max`.
    """
    derivative_shifts = derivative_shifts or {-1: 1.0, 0: -2.0, 1: 1.0}

    data = sp.lil_matrix((n1d_max ** ndim_max, n1d_max ** ndim_max), dtype=float)

    h0_fact = 1 / lattice_spacing ** 2 / 2 / (particle_mass / 2)

    modified_shifts = {
        shift: coeff * h0_fact for shift, coeff in derivative_shifts.items()
    }

    for nr in range(n1d_max ** ndim_max):
        nr_subtracted = nr
        n1d_pow_ndim = 1

        for _ in range(ndim_max):
            nxi = (nr_subtracted % (n1d_pow_ndim * n1d_max)) // n1d_pow_ndim
            nr_subtracted -= nxi * n1d_pow_ndim

            for shift, coeff in modified_shifts.items():
                nr_shift = nr + n1d_pow_ndim * (
                    -nxi + (nxi + shift + n1d_max * 10) % n1d_max
                )
                data[(nr, nr_shift)] -= coeff

            n1d_pow_ndim *= n1d_max

    if cupy_sp and cuda:
        LOGGER.debug("Transfering kinetic hamiltonian to gpu")
        data = cupy_sp.scipy2cupy(data)

    return data.tocsr()


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
        contact_interaction = cupy_sp.scipy2cupy(contact_interaction.tocsr())

    return (contact_interaction + kinetic_hamiltonian).tocsr()


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

    def __post_init__(self):
        self.derivative_shifts = self.derivative_shifts or {-1: 1.0, 0: -2.0, 1: 1.0}
        LOGGER.debug("Allocating kinetic hamiltonian")
        self.kinetic_hamiltonian = get_kinetic_hamiltonian(
            self.n1d_max,
            self.lattice_spacing,
            self.particle_mass,
            self.ndim_max,
            self.derivative_shifts,
            self.cuda,
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
        H = get_full_hamiltonian(
            self.kinetic_hamiltonian,
            contact_strength,
            self.ndim_max,
            self.lattice_spacing,
            self.cuda,
        )
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
        H = get_full_hamiltonian(
            self.kinetic_hamiltonian,
            contact_strength,
            self.ndim_max,
            self.lattice_spacing,
            self.cuda,
        )
        LOGGER.debug("Computing eigenvalues")
        if cupy_sp and self.cuda:
            if "max_iter" not in kwargs:
                kwargs["max_iter"] = n_energies + 10
            out = cupy_sp.lanczos_cp(H, n_eigs=n_energies, **kwargs)
        else:
            out = lina.eigsh(H, k=n_energies, which="SA", **kwargs)[0]
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

    psi0 = np.exp(-kappa * r) / (r + 1.e-7)
    psi0 /= np.sqrt(psi0 @ psi0 * lattice_spacing ** ndim_max)

    return psi0
