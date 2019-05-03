"""Implementation of contact Hamiltonians
"""
from dataclasses import dataclass
from dataclasses import field

import os
import logging

import numpy as np
from scipy.sparse.linalg import eigsh

from luescher_nd.utilities import get_logger

from luescher_nd.hamiltonians.kinetic import MomentumKineticHamiltonian

from luescher_nd.database.connection import DatabaseSession

from luescher_nd.database.tables import create_db
from luescher_nd.database.tables import LongRangeEnergyEntry


LOGGER = get_logger(logging.INFO)

try:
    import cupy as cp  # pylint: disable=E0401

    device_array = cp.ndarray  # pylint: disable=C0103
except ModuleNotFoundError:
    cp = None  # pylint: disable=C0103
    device_array = None  # pylint: disable=C0103


def gamma2gbar(gamma, m, mu):  # pylint: disable=C0103
    """Returns Long range potential normalization depending on binding momentum
    """
    return (
        2
        * (gamma + m) ** 2
        / np.sqrt(mu * m ** 3 * (gamma ** 2 + 5 * m ** 2 + 4 * gamma * m))
        * m
    )


HBARC = 197.327

MPI = 134.98 / HBARC
E0 = -2.225 / HBARC
MN = 937.326 / HBARC
MU = MN / 2

M0 = 20 * MPI

GAMMA0 = np.sqrt(-2 * MU * E0)
GBAR0 = gamma2gbar(GAMMA0, M0, MU)


def p_cot_delta(p, gbar, mu, m):  # pylint: disable=C0103
    """Returns effecive range expansion for long range potential
    """
    d0 = 16 * gbar ** 2 * mu
    res = -(5 * gbar ** 2 * mu * m - 4 * m ** 2) / d0
    res += (15 * gbar ** 2 * mu + 16 * m) / d0 / m * p ** 2
    res += (5 * gbar ** 2 * mu + 24 * m) / d0 / m ** 3 * p ** 4
    res += (gbar ** 2 * mu + 16 * m) / d0 / m ** 5 * p ** 6
    res += 4 * m / d0 / m ** 7 * p ** 8
    return res


@dataclass(frozen=True)
class PhenomLRHamiltonian(MomentumKineticHamiltonian):
    """Phenemenological long range Hamiltonian mimicing two-pion exchange.
    """

    M: float = M0
    gbar: float = GBAR0

    _mat: np.ndarray = field(init=False, repr=False, default=None)
    _V: np.ndarray = field(init=False, repr=False, default=None)
    _gp: np.ndarray = field(init=False, repr=False, default=None)
    _mat_device: device_array = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initializes potential matrix
        """
        super(PhenomLRHamiltonian, self).__post_init__()
        object.__setattr__(
            self,
            "_gp",
            self.gbar * np.sqrt(8 * np.pi) * self.M ** 3 / (self.p2 + self.M ** 2) ** 2,
        )

    @property
    def mat(self):  # pylint: disable=C0103
        """Hamiltonian on CPU (lazyloaded).
        """
        if self._mat is None:
            object.__setattr__(self, "_mat", np.diag(self._disp_over_m) + self.potential)
        return self._mat

    @property
    def mat_device(self):  # pylint: disable=C0103
        """Hamiltonian on CPU (lazyloaded).
        """
        if cp is None:
            raise ModuleNotFoundError("Could not import cupy -- cannot use GPU!")
        if self._mat_device is None:
            gp_device = cp.array(self._gp)
            object.__setattr__(
                self,
                "_mat_device",
                cp.diag(self._disp_over_m) - gp_device.reshape(-1, 1) * gp_device,
            )
        return self._mat_device

    def apply(self, vec):
        """Applies hamiltonian to vector
        """
        return self._disp_over_m * vec - self._gp * (  # pylint: disable = E1130
            self._gp @ vec / self.L ** self.ndim
        )

    @property
    def potential(self) -> np.ndarray:
        """Returns the potential part of the Hamiltonian
        """
        if self._V is None:
            potential = -self._gp.reshape(-1, 1) * self._gp  # pylint: disable=E1101
            object.__setattr__(self, "_V", potential)
        return self._V


def export_eigs(  # pylint: disable=C0103, R0914
    h: PhenomLRHamiltonian, db: str, overwrite: bool = False, **kwargs
):
    """Computes eigenvalues of hamiltonian and writes to db.

    First checks if values are already present. If true, does nothing.
    However, it does not check how many levels are present.
    So if you want to compute more levels, set ``overwrite`` to ``True``.

    **Arguments**
        h: PhenomLRHamiltonian
            Hamiltonian to solve.

        db: str
            Database name for export / import.

        overwrite: bool = False
            Overwrite existing entries.

        **kwargs:
            Arguments fed to ``scipy.sparse.linalg.eigsh``.
            Default are ``return_eigenvectors=False`` and ``which="SA"``.
            These cannot be changed.
    """
    kwargs["return_eigenvectors"] = False
    kwargs["which"] = "SA"
    if "k" not in kwargs:
        kwargs["k"] = 6

    if not os.path.exists(db):
        LOGGER.info("Creating database `%s`", db)
        create_db(db)

    h_keys = {
        "n1d": h.n1d,
        "epsilon": h.epsilon,
        "mu": h.m,
        "m": h.M,
        "gbar": h.gbar,
        "nstep": h.nstep,
    }
    with DatabaseSession(db, commit=False) as sess:
        matches = sess.query(LongRangeEnergyEntry).filter_by(**h_keys).all()
        if matches and not overwrite:
            LOGGER.info("Found %d entries for `%s`. Skip.", len(matches), h)
            return

    LOGGER.info("Exporting eigenvalues of `%s` with arguments:", h)
    LOGGER.info("\tDatabase='%s'", db)
    for key, val in kwargs.items():
        LOGGER.info("\t%s=%s", key, val)

    eigs = np.sort(eigsh(h.op, **kwargs))
    n_created = 0
    with DatabaseSession(db) as sess:
        for nlevel, eig in enumerate(np.sort(eigs)):
            data = {
                "E": eig,
                "n1d": h.n1d,
                "epsilon": h.epsilon,
                "mu": h.m,
                "nlevel": nlevel,
                "m": h.M,
                "gbar": h.gbar,
                "nstep": h.nstep,
            }
            entry, created = LongRangeEnergyEntry.get_or_create(session=sess, **data)
            if created:
                LOGGER.debug("Created `%s`", entry)
                n_created += 1

    LOGGER.info("\t-----")
    LOGGER.info("\tExported %d entries", n_created)
