#!/usr/bin/env python
# pylint: disable=C0103
"""Scripts which export eigenvalues of potential fitted to Finite Volume discrete
ground state to database
"""
import os

from argparse import ArgumentParser

import logging

from itertools import product

from yaml import load

from tqdm import tqdm

from luescher_nd.hamiltonians.contact import MomentumContactHamiltonian

from luescher_nd.operators import get_projector_to_parity
from luescher_nd.operators import get_projector_to_a1g

from luescher_nd.zeta.zeta3d import DispersionZeta3d
from luescher_nd.zeta.zeta3d import Zeta3D
from luescher_nd.zeta.extern.pyzeta import (  # pylint: disable=W0611, E0611
    zeta as spherical_zeta,
)

from luescher_nd.solvers.contact import FitKernel

DB_NAME = (
    "contact-fitted"
    "&a_inv={a_inv:+1.1f}"
    "&zeta={zeta}"
    "&projector={projector}"
    "&n_eigs={n_eigs}"
    ".sqlite"
)

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "data"
    )
)

LOGGER = logging.getLogger("export-contact-energies")

PARSER = ArgumentParser(
    description="Script which exports contact interaction energy levels"
    " fitted to effective range expansion to database"
)
PARSER.add_argument(
    "input", type=str, help="YAML input file which describes the run parameters"
)


def get_input():
    """Runs the argument parser
    """
    args = PARSER.parse_args()
    with open(args.input, "r") as fin:
        pars = load(fin.read())

    LOGGER.info("Input parameters:\n%s", pars)
    return pars


def main():
    """Runs the export script
    """
    pars = get_input()
    export_kwargs = {
        "comment": "potential fitted to FV-FS ground state"
        " determined by first zero of zeta function.\n"
        "Parameters:"
        + "\n\tphysics: {0}".format(pars["physics"])
        + "\n\tzeta: {0}".format(pars["zeta"])
        + "\n\teigenvalues: {0}".format(pars["eigenvalues"])
    }

    database_address = os.path.join(
        ROOT,
        DB_NAME.format(
            zeta=pars["zeta"],
            a_inv=pars["physics"]["a_inv"],
            projector=pars["eigenvalues"]["projector"]["type"],
            n_eigs=pars["eigenvalues"]["solver"]["k"],
        ),
    )

    LOGGER.info("Exporting energies to %s", database_address)

    if pars["zeta"] == "spherical":
        zeta = lambda n1d, epsilon, nstep: spherical_zeta
    elif pars["zeta"] == "dispersion":
        zeta = DispersionZeta3d
    elif pars["zeta"] == "cartesian":
        zeta = lambda n1d, epsilon, nstep: Zeta3D(n1d, epsilon)
    else:
        raise KeyError("Recieved unkwon parater for zeta function (`{zeta_key}`)")

    projector_type = pars["eigenvalues"]["projector"]["type"]
    projector_cutoff = pars["eigenvalues"]["projector"]["cutoff"]
    if projector_type == "a1g":
        get_projector = lambda n1d: get_projector_to_a1g(n1d, ndim=3)
    elif projector_type == "parity":
        get_projector = lambda n1d: get_projector_to_parity(n1d, ndim=3, positive=False)
    else:
        get_projector = lambda n1d: None

    for nstep, L, epsilon in tqdm(
        list(product(*[pars["basis"][key] for key in ["nstep", "L", "epsilon"]]))
    ):
        n1d = int(L / epsilon)
        if n1d > 50:
            continue

        LOGGER.info(
            "Computing eigenvalues for n1d=%d, epsilon=%f, nstep=%s", n1d, epsilon, nstep
        )

        if MomentumContactHamiltonian.exists_in_db(
            database_address, n1d=n1d, epsilon=epsilon, nstep=nstep
        ):
            LOGGER.info("Skipp [entry already exists in db]")
            continue

        h = MomentumContactHamiltonian(
            n1d=n1d,
            epsilon=epsilon,
            nstep=nstep,
            filter_out=get_projector(n1d),
            filter_cutoff=projector_cutoff,
        )

        kernel = FitKernel(
            h, zeta=zeta(n1d, epsilon, nstep), a_inv=pars["physics"]["a_inv"]
        )
        contact_strength = kernel.fit_contact_strenth()
        h.set_contact_strength(contact_strength)

        continue

        eigsh_kwargs = pars["eigenvalues"]["solver"]
        eigsh_kwargs["k"] = (
            eigsh_kwargs["k"] if eigsh_kwargs["k"] < n1d ** 3 - 1 else n1d ** 3 - 2
        )
        h.export_eigs(
            database_address, eigsh_kwargs=eigsh_kwargs, export_kwargs=export_kwargs
        )


if __name__ == "__main__":
    main()
