"""Allocation of Hamiltonian operator
"""
from dataclasses import dataclass
from dataclasses import field

import numpy as np

from scipy.sparse.linalg import LinearOperator

from luescher_nd.utilities import get_laplace_coefficients


@dataclass(frozen=True)
class MomentumContactHamiltonian:
    """Contact interaction Hamiltonian in momentum space
    """

    n1d: int
    epsilon: float = 1.0
    m: float = 4.758
    ndim: int = 3
    c0: float = -1.0
    nstep: int = 3

    _disp_over_m: np.ndarray = field(init=False, repr=False)
    _mat: LinearOperator = field(init=False, repr=False)

    def L(self):  # pylint: disable=C0103
        """Lattice spacing time nodes in one direction
        """
        return self.epsilon * self.n1d

    def __post_init__(self):
        """Initializes the dispersion relation the matvec kernel.
        """
        coeffs = get_laplace_coefficients(self.nstep)
        p1d = np.arange(self.n1d) * 2 * np.pi / self.L
        disp1d = np.sum([-cn * np.cos(n * p1d) for n, cn in coeffs.items()], axis=0)

        disp = np.sum(np.array(np.meshgrid(*[disp1d] * self.ndim)), axis=0).flatten()

        object.__setattr__(self, "_disp_over_m", disp / 2 / self.m)
        object.__setattr__(
            self,
            "_mat",
            LinearOperator(matvec=self.apply, shape=[self.n1d ** self.ndim] * 2),
        )

    @property
    def mat(self) -> LinearOperator:
        """The matvec kernel of the Hamiltonian
        """
        return self._mat

    def apply(self, vec):
        """Applies hamiltonian to vector
        """
        return self._disp_over_m * vec + self.c0 * np.sum(vec) / self.L ** self.ndim
