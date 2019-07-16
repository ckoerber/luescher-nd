"""Utility functions for database submodule
"""
from typing import Dict

from itertools import product

import os

import numpy as np
import pandas as pd

import lsqfit
import gvar as gv

from luescher_nd.database.connection import DatabaseSession

from luescher_nd.hamiltonians.kinetic import HBARC

from luescher_nd.zeta.zeta3d import DispersionZeta3d
from luescher_nd.zeta.extern.pyzeta import zeta  # pylint: disable=E0611

DATA_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
)

np.random.seed(42)


def read_table(  # pylint: disable=R0913, R0914
    database: str,
    round_digits: int = 2,
    dispersion_zeta: bool = True,
    filter_poles: bool = False,
    filter_by_nstates: bool = False,
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

        filter_poles: bool = False
            Automatically filter out states which are close to zeta poles.

        filter_by_nstates: bool = False
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


def _even_poly(epsilon: np.ndarray, p: Dict[str, gv.GVar]):  # pylint: disable=E1101
    """Even polynomial used for fit.

    `y = sum(x_n epsilon**(2n), n=0, n_max)`.

    **Arguments**
        epsilon: np.ndarray
            The lattice spacing to extrapolate.

        p: Dict[str, gv.GVar]
            The prior. Must contain key "x"
    """
    nexp = len(p["x"])
    exps = np.arange(0, nexp * 2, 2)
    out = 0
    for exp, x in zip(exps, p["x"]):
        out += epsilon ** exp * x

    return out


def _group_wise_fit(  # pylint: disable=R0914
    row: pd.Series, n_poly_max: int = 4, delta_x=1.0e-8, include_statistics: bool = True
):
    """Runs a bayesian fit to extrapolate x(epsilon) to zero.

    Requires at least 3 data points, otherwise sets result to None.
    Prior is smallest x as offset and 10 * random noize.

    Picks the best fit determined by logGBF as the main result.

    **Arguments**
        row: pd.Series
            Series containing `epsilon` and `x` columns which are needed for the
            extrapolation.

        n_poly_max: int = 4
            Maximal order of the polynomial used for the spectrum extrapolation.
            The fitter runs fits from 1 to `n_poly_max` even polynomials and picks
            the best one defined by the maximum of the logGBF.

        delta_x:float=1.0e-8
            Approximate error for the x-values.

        include_statistics: bool = True
            Includes fit statistics like chi2/dof or logGBF.
    """
    epsilon, x = row["epsilon"].values, row["x"].values
    error = [delta_x * np.random.uniform(0.5, 2, size=1)[0] for _ in x]
    data = (epsilon, gv.gvar(x, error))

    if len(x) > 2:

        fits = []
        log_gbf = []
        for nexp in range(1, n_poly_max + 1):
            x_init = np.zeros(nexp)
            x_init[0] = x[0]
            x_error = [10 * np.random.uniform(0.5, 2, size=1)[0] for _ in x_init]
            prior = {"x": gv.gvar(x_init, x_error)}

            fit = lsqfit.nonlinear_fit(data, fcn=_even_poly, prior=prior)
            fits.append(fit)
            log_gbf.append(fit.logGBF)

        idx = np.argmax(log_gbf)
        fit = fits[idx]

        if include_statistics:
            out = pd.Series(
                {
                    "x": fit.p["x"][0].mean,
                    "sdev": fit.p["x"][0].mean,
                    "chi2/dof": fit.chi2 / fit.dof,
                    "logGBF": fit.logGBF,
                    "n_poly_max": range(1, n_poly_max + 1)[idx],
                }
            )
        else:
            out = pd.Series({"x": fit.p["x"][0].mean})
    else:
        out = None

    return out


def get_continuum_extrapolation(  # pylint: disable=C0103
    df: pd.DataFrame,
    n_poly_max: int = 4,
    delta_x: float = 1.0e-8,
    include_statistics: bool = True,
) -> pd.DataFrame:
    """Takes a data frame read in by read tables and runs a continuum extrapolation
    for the spectrum.

    The continuum extrapolation is executed by a even polynomial fit up to order
    `n_poly_max`

    **Arguments**
        df: pd.DataFrame
            DataFrame returend by `read_table`.

        n_poly_max: int = 4
            Maximal order of the polynomial used for the spectrum extrapolation.
            The fitter runs fits from 1 to `n_poly_max` even polynomials and picks
            the best one defined by the maximum of the logGBF.

        delta_x:float=1.0e-8
            Approximate error for the x-values.

        include_statistics: bool = True
            Includes fit statistics like chi2/dof or logGBF.
    """
    group = df.groupby(["L", "nstep", "nlevel"])[["epsilon", "x"]]
    fit_df = group.apply(
        _group_wise_fit,
        n_poly_max=n_poly_max,
        delta_x=delta_x,
        include_statistics=include_statistics,
    ).reset_index()
    fit_df["epsilon"] = 0

    return fit_df
