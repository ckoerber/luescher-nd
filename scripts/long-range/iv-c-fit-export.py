#!/usr/bin/env python
# pylint: disable=C0103
"""Scripts which export eigenvalues of potential fitted to Infinite Volume Continuum
ground state to database
"""
from dataclasses import dataclass

import os

from itertools import product

from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar

from luescher_nd.hamiltonians.longrange import PhenomLRHamiltonian
from luescher_nd.hamiltonians.longrange import E0

RANGES = {"n1d": range(10, 51, 5), "L": [10.0, 15.0, 20.0], "nstep": [1, 2, 3, 4, None]}
PARS = {"k": 300}

DBNAME = "db-lr-iv-c-fitted.sqlite"

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
)
DB = os.path.join(ROOT, "data", DBNAME)


@dataclass
class FitKernel:
    """Kernel for eigenvalue optimization
    """

    h: PhenomLRHamiltonian
    e0: float = E0

    def chi2(self, gbar):
        """Computes the first eigenvalue and returns the squared difference between with
        expected value.
        """
        hnew = PhenomLRHamiltonian(
            n1d=self.h.n1d, epsilon=self.h.epsilon, nstep=self.h.nstep, gbar=gbar
        )
        e0 = eigsh(hnew.op, which="SA", return_eigenvectors=False, k=1)[0]
        return (e0 - self.e0) ** 2


def main():
    """Runs the export script
    """
    for n1d, L, nstep in product(RANGES["n1d"], RANGES["L"], RANGES["nstep"]):
        epsilon = L / n1d
        h = PhenomLRHamiltonian(n1d=n1d, epsilon=epsilon, nstep=nstep)

        kernel = FitKernel(h)
        gbar = minimize_scalar(kernel.chi2, bracket=(1.0e-4, 1.0e2)).x

        PhenomLRHamiltonian(
            n1d=n1d, epsilon=epsilon, nstep=nstep, gbar=gbar
        ).export_eigs(
            DB,
            eigsh_kwargs=PARS,
            export_kwargs={
                "comment": "potential fitted to Infinite Volume continuum ground state"
            },
        )


if __name__ == "__main__":
    main()
