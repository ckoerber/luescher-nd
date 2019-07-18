#!/usr/bin/env python3
# pylint: disable=C0103, E0611
"""Scripts which export eigenvalues of potential fitted to Finite Volume discrete
ground state to database
"""
from typing import Callable
from typing import Optional

import os

from argparse import ArgumentParser

from json import dumps
import logging
from datetime import datetime

from itertools import product

from yaml import load

import numpy as np

from tqdm import tqdm

from luescher_nd.hamiltonians.contact import MomentumContactHamiltonian

from luescher_nd.operators import get_projector_to_parity
from luescher_nd.operators import get_projector_to_not_a1g

from luescher_nd.solvers.contact import FitKernel

DB_NAME = (
    "contact-fitted"
    "_a-inv={a_inv:+1.1f}"
    "_zeta={zeta}"
    "_projector={projector}"
    "_n-eigs={n_eigs}"
    ".sqlite"
)

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "data"
    )
)


def get_logger() -> logging.Logger:
    """Returns file logger
    """
    logger = logging.getLogger("export-contact-energies")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        "export-contact-energies-%s.log" % datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "a",
    )
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(name)s|%(asctime)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


LOGGER = get_logger()

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

    LOGGER.info("Input parameters:\n%s", str(dumps(pars, indent=4)))
    return pars


def get_zeta(
    kind: str, new: bool = False, N: Optional[int] = None, improved: bool = False
) -> Callable[[int, float, int], [np.ndarray, np.ndarray]]:
    """Returns zeta function of given kind which takes `x` as an argument.
    """
    if not new:
        if kind == "spherical":
            from luescher_nd.zeta.extern.pyzeta import zeta as spherical_zeta

            zeta = lambda n1d, epsilon, nstep: spherical_zeta
        elif kind == "dispersion":
            from luescher_nd.zeta.zeta3d import DispersionZeta3d

            zeta = DispersionZeta3d
        elif kind == "cartesian":
            from luescher_nd.zeta.zeta3d import Zeta3D

            zeta = lambda n1d, epsilon, nstep: Zeta3D(n1d, spherical=False)
        else:
            raise KeyError("Recieved unkwon parater for zeta function (`{kind}`)")
    else:
        if kind == "spherical":
            from luescher_nd.zeta.cpp.pyzeta import SphericalZeta

            zeta = lambda n1d, epsilon, nstep: SphericalZeta(3, N, improved)
        elif kind == "dispersion":
            from luescher_nd.zeta.cpp.pyzeta import DispersionZeta3d

            zeta = lambda n1d, epsilon, nstep: DispersionZeta3d(
                3, n1d, epsilon * n1d, nstep, improved
            )
        elif kind == "cartesian":
            from luescher_nd.zeta.cpp.pyzeta import DispersionZeta3d

            zeta = lambda n1d, epsilon, nstep: Zeta3D(3, N, improved)
        else:
            raise KeyError("Recieved unkwon parater for zeta function (`{kind}`)")

    return zeta


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

    zeta = get_zeta(**pars["zeta"])

    projector_type = pars["eigenvalues"]["projector"]["type"]
    projector_cutoff = pars["eigenvalues"]["projector"]["cutoff"]
    if projector_type == "a1g":
        get_projector = lambda n1d: get_projector_to_not_a1g(n1d, ndim=3)
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
            "Computing eigenvalues for n1d=%02d, epsilon=%1.4f, nstep=%s",
            n1d,
            epsilon,
            nstep,
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

        eigsh_kwargs = pars["eigenvalues"]["solver"]
        eigsh_kwargs["k"] = (
            eigsh_kwargs["k"] if eigsh_kwargs["k"] < n1d ** 3 - 1 else n1d ** 3 - 2
        )
        h.export_eigs(
            database_address, eigsh_kwargs=eigsh_kwargs, export_kwargs=export_kwargs
        )


if __name__ == "__main__":
    main()
