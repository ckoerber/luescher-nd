#!/usr/bin/env python3
# pylint: disable=C0103, E0611
"""Scripts which computes eigenvalues and export overlap
"""
from typing import Dict
from typing import Tuple

import os

import numpy as np
import pandas as pd

from scipy.sparse.linalg import eigsh

from tqdm import tqdm

from luescher_nd.database import utilities as ut
from luescher_nd.database.utilities import DATA_FOLDER

from luescher_nd.operators.a1g import projector

from luescher_nd.operators import get_projector_to_not_a1g
from luescher_nd.database.utilities import get_degeneracy

from luescher_nd.hamiltonians.contact import MomentumContactHamiltonian


FILE_NAME = "contact-fitted_a-inv=+0.0_zeta=spherical_projector=a1g_n-eigs=200.sqlite"
OUT_NAME = FILE_NAME.replace("contact-fitted", "eigs-overlap-nstep={nstep:+d}").replace(
    ".sqlite", ".h5"
)

NDIM = 3
NSTEP_RANGE = [1, 4]
N1D_RANGE = [4, 10, 20, 30, 40, 50]
NLEVEL_MAX = 20


def get_a1g_basis(n1d: int, ndim: int = 3) -> Dict[Tuple[int, int, int], np.ndarray]:
    """Gets a1g basis.

    Returns dictinary where keys are tuples corresponding to a1g momentum equivalence
    classes and values are vectors in the full momentum space.

    E.g. `{(0, 0, 0): [1, 0, 0, ....]}`

    **Arguments**
        n1d: int
            Number of spatial nodes in one direction.

        ndim: int = 3
            Number of spatial dimensions
    """
    degs = get_degeneracy(n1d, ndim)
    p = projector(n1d, ndim)
    basis = {}
    for vec_set in degs.values():
        for vec in vec_set:
            pvec = (n1d ** np.arange(ndim)) @ np.array(vec)
            bvec = p.T[pvec].toarray().flatten()
            bvec /= np.sqrt(bvec @ bvec)
            basis[vec] = bvec
    return basis


def main():  # pylint: disable=R0914
    """Recomputes first 20 eigenstates of contact interaction at unitarity and computes
    the overlap of eigenstates with eigenstates of a1g.

    Results are exported ot hdf5 file.
    """

    data = []
    for nstep in NSTEP_RANGE:
        file_name = os.path.join(DATA_FOLDER, OUT_NAME.format(nstep=nstep))
        print(f"[+] exporting nstep = {nstep} data to {file_name}")

        df = ut.read_table(
            os.path.join(DATA_FOLDER, FILE_NAME),
            zeta=None,
            round_digits=2,
            filter_poles=False,
            filter_by_nstates=False,
            filter_degeneracy=False,
        ).query("nlevel == 0 and nstep == @nstep and L == 1")[
            ["n1d", "epsilon", "nstep", "L", "x", "nlevel", "contact_strength", "E"]
        ]

        interactions = {
            (n1d, epsilon): c0
            for (n1d, epsilon), c0 in df.set_index(["n1d", "epsilon"])[
                "contact_strength"
            ]
            .to_dict()
            .items()
            if n1d % 2 == 0 and n1d in N1D_RANGE
        }

        data = []
        for (n1d, epsilon), c0 in tqdm(interactions.items()):

            pnot = get_projector_to_not_a1g(n1d, NDIM)

            basis = get_a1g_basis(n1d, ndim=NDIM)

            h = MomentumContactHamiltonian(
                n1d,
                epsilon=epsilon,
                ndim=NDIM,
                nstep=None,
                contact_strength=c0,
                filter_out=pnot,
                filter_cutoff=3.0e2,
            )
            E, v = eigsh(h.op, k=NLEVEL_MAX, which="SA", tol=1.0e-16)
            x = 2 * h.mass / 2 * E * h.L ** 2 / 4 / np.pi ** 2

            for nlevel, (xx, vv) in enumerate(zip(x, v.T)):
                for key, bv in basis.items():
                    coeff = bv @ vv
                    overlap = coeff ** 2

                    if overlap > 1.0e-4:
                        data.append(
                            {
                                "n1d": n1d,
                                "epsilon": epsilon,
                                "L": epsilon * n1d,
                                "x": xx,
                                "nlevel": nlevel,
                                "overlap": overlap,
                                "coeff": coeff,
                                "a1g": str(key),
                                "nstep": nstep,
                                "contact_strength": c0,
                            }
                        )

            pd.DataFrame(data).to_hdf(file_name, key="overlap")


if __name__ == "__main__":
    main()
