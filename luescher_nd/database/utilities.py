"""Utility functions for database submodule
"""
from itertools import product

import os

import numpy as np
import pandas as pd

from luescher_nd.database.connection import DatabaseSession

from luescher_nd.hamiltonians.kinetic import HBARC

from luescher_nd.zeta.zeta3d import DispersionZeta3d
from luescher_nd.zeta.extern.pyzeta import zeta  # pylint: disable=E0611

DATA_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
)


def read_table(
    database: str,
    round_digits: int = 2,
    dispersion_zeta: bool = True,
    filter_poles: bool = True,
    filter_by_nstates: bool = True,
    drop_comment: bool = True,
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

        filter_by_nstates: bool = True
            Automatically filter out states which are degenerate.

        drop_comment: bool = True
            Drops the comment column.
    """
    sess = DatabaseSession(database, commit=False)
    df = pd.read_sql_table("energy", sess.engine, index_col="id")
    energy_type = df["type"].unique()

    if len(energy_type) == 1:
        df = pd.merge(
            df,
            pd.read_sql_table(energy_type[0], sess.engine, index_col="id"),
            left_index=True,
            right_index=True,
        )

    if drop_comment:
        df = df.drop(columns=["comment"])

    df["L"] = df["n1d"] * df["epsilon"]
    df["nstep"] = df["nstep"].fillna(-1).astype(int)
    df["E [MeV]"] = df["E"] * HBARC
    df["p2"] = df["E"] * 2 * df["mass"] / 2
    df["x"] = df["p2"] / (2 * np.pi / df["L"]) ** 2

    if filter_by_nstates:
        group = (
            df.reset_index()
            .round(round_digits)
            .groupby(["epsilon", "L", "nstep", "x"])[["nlevel", "id"]]
        )
        grouped = group.agg({"nlevel": "count", "id": "mean"}).rename(
            columns={"nlevel": "count"}
        )
        count_one = grouped[grouped["count"] == 1].astype({"id": int}).set_index("id")
        df = pd.merge(df, count_one, how="inner", on=["id"])

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
        z = (
            DispersionZeta3d(L, epsilon, nstep if nstep > 0 else None)
            if dispersion_zeta
            else zeta
        )
        df.loc[ind, "y"] = z(df.loc[ind, "x"].values) / np.pi / df.loc[ind, "L"]

    if filter_poles:
        df = df.dropna()

    df.loc[df.nstep == -1, "nstep"] = r"$\infty$"
    df["nstep"] = df["nstep"].astype(str)

    return df
