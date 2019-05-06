#!/usr/bin/env python
# pylint: disable=C0103
"""Scripts which export eigenvalues of potential fixed with Infinite Volume Continuum
parameters to database
"""
import os

from itertools import product

from luescher_nd.hamiltonians.longrange import PhenomLRHamiltonian
from luescher_nd.hamiltonians.longrange import export_eigs

RANGES = {"n1d": range(10, 51, 5), "L": [10.0, 15.0, 20.0], "nstep": [1, 2, 3, 4, None]}
PARS = {"k": 300}

DBNAME = "db-lr-iv-c-fixed.sqlite"

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
)
DB = os.path.join(ROOT, "data", DBNAME)


def main():
    """Runs the export script
    """
    for n1d, L, nstep in product(RANGES["n1d"], RANGES["L"], RANGES["nstep"]):
        epsilon = L / n1d
        h = PhenomLRHamiltonian(n1d=n1d, epsilon=epsilon, nstep=nstep)
        export_eigs(h, DB, **PARS)


if __name__ == "__main__":
    main()
