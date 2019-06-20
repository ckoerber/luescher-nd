"""Utility functions for database submodule
"""
from itertools import product

import numpy as np
import pandas as pd

from luescher_nd.database.connection import DatabaseSession

from luescher_nd.hamiltonians.kinetic import HBARC

from luescher_nd.zeta.zeta3d import Zeta3D
from luescher_nd.zeta.extern.pyzeta import zeta  # pylint: disable=E0611


def read_table(
    database: str,
    round_digits: int = 2,
    dispersion_zeta: bool = True,
    filter_poles: bool = True,
) -> pd.DataFrame:
    """Reads database and returns data frame with table plus additional columns.

    The database must contain an ``Energy`` table.
    Drops all entries which are at or close to divergencies of the zeta function.

    **Argumennts**
        database: str
            The sqlite database file.

        round_digits: int = 2
            How close an x-value has to be to an integer in order to be dropped.

        dispersion_zeta: bool = True
            Use dispersion zeta function to compute the ``y`` column.
            Else use regular zeta.

        filter_poles: bool = True
            Automatically filter out states which are close to zeta poles.
    """
    sess = DatabaseSession(database, commit=False)
    df = pd.read_sql_table("energy", sess.engine, index_col="id").drop(
        columns=["comment"]
    )

    df["L"] = df["n1d"] * df["epsilon"]
    df["nstep"] = df["nstep"].fillna(-1).astype(int)
    df["E [MeV]"] = df["E"] * HBARC
    df["p2"] = df["E"] * 2 * df["mass"] / 2
    df["x"] = df["p2"] / (2 * np.pi / df["L"]) ** 2

    if filter_poles:
        integer_x = [
            x for x in range(int(df.x.min()) - 1, int(df.x.max() + 2)) if x != 0
        ]
        df = df[~df.x.round(round_digits).isin(integer_x)]

    for L, n1d, nstep in list(
        product(df.L.unique(), df.n1d.unique(), df.nstep.unique())
    ):
        epsilon = L / n1d
        ind = (df.L == L) & (df.n1d == n1d) & (df.nstep == nstep)
        z = Zeta3D(L, epsilon, nstep if nstep > 0 else None) if dispersion_zeta else zeta
        df.loc[ind, "y"] = z(df.loc[ind, "x"].values) / np.pi / df.loc[ind, "L"]

    if filter_poles:
        df = df.dropna()

    df.loc[df.nstep == -1, "nstep"] = r"$\infty$"
    df["nstep"] = df["nstep"].astype(str)

    return df
