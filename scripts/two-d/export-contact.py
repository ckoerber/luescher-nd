#!/usr/bin/env python
# pylint: disable=C0103
"""Scripts which export eigenvalues of potential fitted to Finite Volume continuum
first excited state to database
"""
from typing import Callable

from dataclasses import dataclass
from dataclasses import field

import os

from itertools import product

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar

from luescher_nd.hamiltonians.contact import MomentumContactHamiltonian
from luescher_nd.operators import get_parity_projector

from luescher_nd.zeta.zeta2d import Zeta2D

RANGES = {
    "n1d": [10, 20, 30, 40, 50],
    "L": [1.0, 2.0],
    "nstep": [1, 4, None],
    "spherical": [True, False],
}
PARS = {"k": 50}

NDIM = 2
NMAX = 40
A0 = 1.0

DBNAME = "db-contact-fv-c-parity.sqlite"

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
)
DB = os.path.join(ROOT, "data", "two-d", DBNAME)


@dataclass
class FitKernel:
    """Kernel for eigenvalue optimization
    """

    h: MomentumContactHamiltonian
    zeta: Callable[[np.ndarray], np.ndarray]
    a0: float
    _e1: float = field(init=False, repr=False)

    def __post_init__(self):
        """Init the zeta function"""
        self._e1 = self._get_first_state()

    def _ere_zeta(self, x: np.ndarray) -> np.ndarray:
        r"""Computes the effective range expansion using the 2-d zeta function

        $$ \\frac{1}{\\pi^2} S_2(x) + \\frac{2}{\\pi} \\log(\\sqrt{x})$$

        The variable $$x$$$ is related to $$p$$ by

        $$ x = \\right(\\frac{p L}{2 \\pi}\\left)^2$$
        """
        return self.zeta(x) / np.pi ** 2 + 1 / np.pi * np.log(x)

    def _ere_analytic(self, x: np.ndarray) -> np.ndarray:
        r"""Computes the FV continuum effective range expansion

        $$ \\frac{2}{\\pi} \\log(p a_0)$$

        where $a_0$ is the scattering length.

        The variable $$x$$$ is related to $$p$$ by

        $$ x = \\right(\\frac{p L}{2 \\pi}\\left)^2$$
        """
        return 1 / np.pi * np.log(x) + 2 / np.pi * np.log(2 * np.pi * self.a0 / self.h.L)

    def _ere_diff_sqrd(self, x: float) -> float:
        """Computes the difference betewen ERE and zeta function
        """
        return (self._ere_analytic(x) - self._ere_zeta(x)) ** 2

    def _get_first_state(self) -> float:
        """Computes the first intersection of the zeta function and the ERE
        """
        x0 = minimize_scalar(
            self._ere_diff_sqrd, bracket=(0, 0.5), tol=1.0e-12, options={"xtol": 1.0e-12}
        ).x[0]
        return x0 / 2 / (self.h.mass / 2) * (2 * np.pi / self.h.L) ** 2

    def chi2(self, contact_strength: float) -> float:
        """Computes the first eigenvalue and returns the squared difference between with
        expected value.
        """
        hnew = self.h.__class__(
            n1d=self.h.n1d,
            epsilon=self.h.epsilon,
            nstep=self.h.nstep,
            mass=self.h.mass,
            contact_strength=-np.exp(contact_strength),
            filter_out=self.h.filter_out,
            filter_cutoff=self.h.filter_cutoff,
            ndim=NDIM,
        )
        e1 = np.sort(eigsh(hnew.op, which="SA", return_eigenvectors=False, k=5))[1]
        return (e1 - self._e1) ** 2


def main():
    """Runs the export script
    """
    for n1d, L, nstep in product(RANGES["n1d"], RANGES["L"], RANGES["nstep"]):
        epsilon = L / n1d
        p_minus = get_parity_projector(n1d, ndim=NDIM, positive=False)
        h = MomentumContactHamiltonian(
            n1d=n1d,
            epsilon=epsilon,
            nstep=nstep,
            filter_out=p_minus,
            filter_cutoff=1.0e2,
            ndim=NDIM,
        )
        for spherical in RANGES["spherical"]:
            zeta = Zeta2D(N=NMAX, spherical=spherical)

            kernel = FitKernel(h, zeta, a0=A0)
            contact_strength = minimize_scalar(
                kernel.chi2, options={"xtol": 1.0e-12}, tol=1.0e-12
            ).x

            MomentumContactHamiltonian(
                n1d=n1d,
                epsilon=epsilon,
                nstep=nstep,
                contact_strength=-np.exp(contact_strength),
                filter_out=p_minus,
                filter_cutoff=1.0e2,
                ndim=NDIM,
            ).export_eigs(
                DB,
                eigsh_kwargs=PARS,
                export_kwargs={
                    "comment": f"spherical={spherical:b}&a0={A0:1.4f}&n={NMAX}"
                },
            )


if __name__ == "__main__":
    main()
