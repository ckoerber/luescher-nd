#!/usr/bin/env python
# pylint: disable=C0103
"""Scripts which export eigenvalues of potential fitted to Finite Volume continuum
ground state to database
"""
from typing import Callable

from dataclasses import dataclass
from dataclasses import field

import os

from itertools import product

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar

from luescher_nd.hamiltonians.longrange import PhenomLRHamiltonian
from luescher_nd.hamiltonians.longrange import export_eigs
from luescher_nd.hamiltonians.longrange import p_cot_delta
from luescher_nd.hamiltonians.longrange import GBAR0, M0, MU

from luescher_nd.zeta.extern.pyzeta import zeta

RANGES = {"n1d": range(10, 51, 5), "L": [10.0, 15.0, 20.0], "nstep": [1, 2, 3, 4, None]}
PARS = {"k": 300}

DBNAME = "db-lr-fv-c-fitted.sqlite"

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
)
DB = os.path.join(ROOT, "data", DBNAME)


@dataclass
class FitKernel:
    """Kernel for eigenvalue optimization
    """

    h: PhenomLRHamiltonian
    _zeta: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)
    _e0: float = field(init=False, repr=False)

    def __post_init__(self):
        """Init the zeta function"""
        self._zeta = zeta
        self._e0 = self._get_ground_state()

    def zeta(self, x: np.ndarray) -> np.ndarray:
        """Lattice zeta function for init parameters.
        """
        return self._zeta(x)

    def _ere_diff_sqrd(self, x: float) -> float:
        """Computes the difference betewen ERE and zeta function
        """
        p = np.sqrt(x + 0j) * 2 * np.pi / self.h.L
        return (
            p_cot_delta(p, gbar=GBAR0, mu=MU, m=M0).real
            - self.zeta(x)[0] / np.pi / self.h.L
        ) ** 2

    def _get_ground_state(self) -> float:
        """Computes the first intersection of the zeta function and the ERE
        """
        x0 = minimize_scalar(self._ere_diff_sqrd, bracket=(-1.0e1, -1.0e-2)).x
        return x0 / 2 / MU * (2 * np.pi / self.h.L) ** 2

    def chi2(self, gbar: float) -> float:
        """Computes the first eigenvalue and returns the squared difference between with
        expected value.
        """
        hnew = PhenomLRHamiltonian(
            n1d=self.h.n1d, epsilon=self.h.epsilon, nstep=self.h.nstep, gbar=gbar
        )
        e0 = eigsh(hnew.op, which="SA", return_eigenvectors=False, k=1)[0]
        return (e0 - self._e0) ** 2


def main():
    """Runs the export script
    """
    for n1d, L, nstep in product(RANGES["n1d"], RANGES["L"], RANGES["nstep"]):
        epsilon = L / n1d
        h = PhenomLRHamiltonian(n1d=n1d, epsilon=epsilon, nstep=nstep)

        kernel = FitKernel(h)
        gbar = minimize_scalar(kernel.chi2, bracket=(1.0e-4, 1.0e2)).x

        h = PhenomLRHamiltonian(n1d=n1d, epsilon=epsilon, nstep=nstep, gbar=gbar)
        export_eigs(h, DB, **PARS)


if __name__ == "__main__":
    main()
