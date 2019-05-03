"""Implementation of kinetic Hamiltonians
"""
from typing import Optional
from typing import Dict

from dataclasses import dataclass
from dataclasses import field

import logging

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import LinearOperator

try:
    from solvers.src import cupy_sp
    import cupy as cp  # pylint: disable=E0401, W0611
except ModuleNotFoundError:
    cupy_sp = None  # pylint: disable=C0103

from luescher_nd.utilities import get_logger
from luescher_nd.utilities import get_laplace_coefficients

LOGGER = get_logger(logging.INFO)


def get_kinetic_hamiltonian(  # pylint: disable=R0914, R0913
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
        data = cupy_sp.scipy2cupy(data)  # pylint: disable=E1101

    return data.tocsr()


@dataclass(frozen=True)
class MomentumKineticHamiltonian:
    """Kinetic Hamiltonian in momentum space
    """

    n1d: int
    epsilon: float = 1.0
    m: float = 4.758
    ndim: int = 3
    nstep: Optional[int] = 3

    _disp_over_m: np.ndarray = field(init=False, repr=False)
    _op: LinearOperator = field(default=None, repr=False)

    @property
    def L(self):  # pylint: disable=C0103
        """Lattice spacing time nodes in one direction
        """
        return self.epsilon * self.n1d

    @property
    def p2(self):  # pylint: disable=C0103
        """Returns momentum dispersion
        """
        return self._disp_over_m * 2 * (self.m / 2)

    def __post_init__(self):
        """Initializes the dispersion relation the matvec kernel.
        """
        if self.nstep is None:
            p1d = (
                np.array(
                    list(range(0, (self.n1d + 1) // 2))
                    + list(range(self.n1d // 2, 0, -1))
                )
                * 2
                * np.pi
                / self.L
            )
            disp1d = p1d ** 2
        else:
            coeffs = get_laplace_coefficients(self.nstep)
            p1d = np.arange(self.n1d) * 2 * np.pi / self.L
            disp1d = np.sum(
                [
                    -cn * np.cos(n * p1d * self.epsilon) / self.epsilon ** 2
                    for n, cn in coeffs.items()
                ],
                axis=0,
            )

        disp = np.sum(np.array(np.meshgrid(*[disp1d] * self.ndim)), axis=0).flatten()

        object.__setattr__(self, "_disp_over_m", disp / 2 / (self.m / 2))

    @property
    def op(self) -> LinearOperator:  # pylint: disable=C0103
        """The matvec kernel of the Hamiltonian
        """
        if not self._op:
            object.__setattr__(
                self,
                "_op",
                LinearOperator(matvec=self.apply, shape=[self.n1d ** self.ndim] * 2),
            )
        return self._op

    def apply(self, vec):
        """Applies hamiltonian to vector
        """
        return self._disp_over_m * vec
