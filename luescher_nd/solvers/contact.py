"""Interface for fitting LEC of contact interaction to effective range
"""
from typing import Callable
from typing import Tuple

from dataclasses import dataclass
from dataclasses import field

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar

from luescher_nd.hamiltonians.contact import MomentumContactHamiltonian


@dataclass
class FitKernel:
    """Kernel for eigenvalue optimization
    """

    h: MomentumContactHamiltonian
    zeta: Callable[[np.ndarray], np.ndarray]
    a_inv: float = 0
    xtol: float = 1.0e-16
    _e0: float = field(init=False, repr=False)

    def __post_init__(self):
        """Init the zeta function"""
        self._e0 = self._get_ground_state()

    def _ere_diff_sqrd(self, x: float) -> float:
        """Computes the difference betewen ERE and zeta function
        """
        return (self.a_inv - self.zeta(x)[0] / np.pi / self.h.L) ** 2

    def _get_ground_state(self) -> float:
        """Computes the first intersection of the zeta function and the ERE
        """
        x0 = minimize_scalar(
            self._ere_diff_sqrd, bracket=(-1.0e1, -1.0e-2), options={"xtol": self.xtol}
        ).x
        return x0 / 2 / (self.h.mass / 2) * (2 * np.pi / self.h.L) ** 2

    def get_zeta_intersection(self, bounds: Tuple[float], **kwargs) -> float:
        """Computes the intersection of the zeta function and the ERE for given bracket
        """
        options = kwargs.pop("options", {"xtol": self.xtol})
        method = kwargs.pop("method", "bounded")
        x = minimize_scalar(
            self._ere_diff_sqrd, bounds=bounds, options=options, method=method, **kwargs
        ).x
        return x

    def chi2(self, contact_strength: float) -> float:
        """Computes the first eigenvalue and returns the squared difference between with
        expected value.
        """
        hnew = self.h.__class__(
            n1d=self.h.n1d,
            epsilon=self.h.epsilon,
            nstep=self.h.nstep,
            mass=self.h.mass,
            contact_strength=contact_strength,
        )
        e0 = eigsh(hnew.op, which="SA", return_eigenvectors=False, k=1)[0]
        return (e0 - self._e0) ** 2

    def fit_contact_strenth(
        self, bracket: Tuple[float, float] = (-1.0e2, -1.0e-4)
    ) -> float:
        """Fits the interaction to match the first excited state of the zeta function

        **Arguments**
            bracket: Tuple[float, float] = (-1.0e2, -1.0e-4)
                The search interval for the solver (see `scipy.optimize.minimize_scalar`)
        """
        return minimize_scalar(self.chi2, bracket=bracket, options={"xtol": self.xtol}).x
