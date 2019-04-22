"""Implementation of contact Hamiltonians
"""
from dataclasses import dataclass
from dataclasses import field

import logging

import numpy as np


from luescher_nd.utilities import get_logger

from luescher_nd.hamiltonians.kinetic import MomentumKineticHamiltonian

LOGGER = get_logger(logging.INFO)


@dataclass(frozen=True)
class PhenomLRHamiltonian(MomentumKineticHamiltonian):
    """Phenemenological long range Hamiltonian mimicing two-pion exchange.
    """

    M: float = 0.1438
    gbar: float = 0.8945

    _H: np.ndarray = field(init=False, repr=False)
    _V: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Initializes potential matrix
        """
        super(PhenomLRHamiltonian, self).__post_init__()
        gp = self.gbar * np.sqrt(8 * np.pi) * self.M ** 3 / (self.p2 + self.M ** 2) ** 2
        potential = -gp.reshape(-1, 1) * gp
        object.__setattr__(self, "_V", potential)
        object.__setattr__(self, "_H", np.diag(self._disp_over_m) + potential)

    def apply(self, vec):
        """Applies hamiltonian to vector
        """
        return self._H @ vec

    @property
    def potential(self) -> np.ndarray:
        """Returns the potential part of the Hamiltonian
        """
        return self._V
