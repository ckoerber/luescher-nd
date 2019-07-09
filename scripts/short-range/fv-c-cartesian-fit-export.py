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

from luescher_nd.hamiltonians.contact import MomentumContactHamiltonian
from luescher_nd.operators import get_parity_projector

from luescher_nd.zeta.zeta3d import Zeta3D  # pylint: disable=E0611

RANGES = {"epsilon": [0.25, 0.2, 0.1, 0.05, 0.025, 0.020], "L": [1.0], "nstep": [1]}


PARS = {"k": 50}

A_INV = -5.0

DBNAME = "db-contact-fv-c-cartesian-fitted-parity-a-inv.sqlite"

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
)
DB = os.path.join(ROOT, "data", DBNAME)


SPHERICAL = False
N = 200
LAMBDA = 2 * np.pi * N


@dataclass
class FitKernel:
    """Kernel for eigenvalue optimization
    """

    h: MomentumContactHamiltonian
    a_inv: float = 0
    _zeta: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)
    _e0: float = field(init=False, repr=False)

    def __post_init__(self):
        """Init the zeta function"""
        self._zeta = Zeta3D(N=LAMBDA / np.pi / 2 / self.h.L)
        self._e0 = self._get_ground_state()

    def zeta(self, x: np.ndarray) -> np.ndarray:
        """Lattice zeta function for init parameters.
        """
        return self._zeta(x)

    def _ere_diff_sqrd(self, x: float) -> float:
        """Computes the difference betewen ERE and zeta function
        """
        return (self.a_inv - self.zeta(x)[0] / np.pi / self.h.L) ** 2

    def _get_ground_state(self) -> float:
        """Computes the first intersection of the zeta function and the ERE
        """
        x0 = minimize_scalar(
            self._ere_diff_sqrd, bracket=(-1.0e1, -1.0e-2), options={"xtol": 1.0e-12}
        ).x
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
            contact_strength=contact_strength,
        )
        e0 = eigsh(hnew.op, which="SA", return_eigenvectors=False, k=1)[0]
        return (e0 - self._e0) ** 2


def main():
    """Runs the export script
    """
    for epsilon, L, nstep in product(RANGES["epsilon"], RANGES["L"], RANGES["nstep"]):
        n1d = int(L / epsilon)
        p_minus = get_parity_projector(n1d, ndim=3, positive=False)
        h = MomentumContactHamiltonian(
            n1d=n1d,
            epsilon=epsilon,
            nstep=nstep,
            filter_out=p_minus,
            filter_cutoff=1.0e2,
        )

        kernel = FitKernel(h, a_inv=A_INV)

        contact_strength = minimize_scalar(
            kernel.chi2, bracket=(-1.0e2, -1.0e-4), options={"xtol": 1.0e-12}
        ).x

        MomentumContactHamiltonian(
            n1d=n1d,
            epsilon=epsilon,
            nstep=nstep,
            contact_strength=contact_strength,
            filter_out=p_minus,
            filter_cutoff=1.0e2,
        ).export_eigs(
            DB,
            eigsh_kwargs=PARS,
            export_kwargs={
                "comment": "potential fitted to Finite Volume"
                " discrete ground state "
                f"(a_inv={A_INV}, Lambda={LAMBDA}, spherical={SPHERICAL})"
            },
        )


if __name__ == "__main__":
    main()