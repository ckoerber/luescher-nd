"""Utility functions for database submodule
"""
from typing import Dict
from typing import Optional
from typing import List
from typing import Tuple

import os

import numpy as np
import pandas as pd

try:
    import lsqfit
    import gvar as gv
except:
    lsqfit = None
    gv = None
    print("Failed to import `lsqfit` and `gvar`. Fitting not possible.")

from luescher_nd.database.connection import DatabaseSession

from luescher_nd.hamiltonians.kinetic import HBARC

from luescher_nd.zeta.zeta3d import DispersionZeta3D
from luescher_nd.zeta.zeta3d import Zeta3D
from luescher_nd.zeta.extern.pyzeta import (  # pylint: disable=E0611
    zeta as spherical_zeta,
)

DATA_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
)

np.random.seed(42)


def get_degeneracy(n1d: int, ndim: int = 3) -> Dict[int, List[Tuple[int, int, int]]]:
    """Computes n@n from lattice vectors in [-n1d//2, n1d//2) and counts degeneracy.

    Vectors are gegenerate if there is no permutation such that abs(v1) == abs(v2) but
    still v1@v1 == v2@v2.

    The dictionary contains equivalency classes for each v@v value.

    **Arguments**
        n1d: int
            Number of lattice sides in on dimension.

        ndim: int = 3
            Number of dimensions.
    """
    n1d = np.arange(-n1d // 2, n1d // 2)

    tmp = {}
    for n in np.transpose([el.flatten() for el in np.meshgrid(*[n1d] * ndim)]):
        n2 = n @ n
        tmp.setdefault(n2, set()).add(tuple(sorted(np.abs(n))))

    n_squared = {key: tmp[key] for key in np.sort(list(tmp.keys()))}

    return n_squared


def get_degeneracy_list(n1d: int, ndim: int = 3) -> List[int]:
    """Computes n@n from lattice vectors in [-n1d//2, n1d//2) and counts degeneracy.

    Vectors are gegenerate if there is no permutation such that abs(v1) == abs(v2) but
    still v1@v1 == v2@v2.

    The returned list returns the n**2 values which are degenerate.

    **Arguments**
        n1d: int
            Number of lattice sides in on dimension.

        ndim: int = 3
            Number of dimensions.
    """
    return [n2 for n2, vecs in get_degeneracy(n1d, ndim=ndim).items() if len(vecs) > 1]


def read_table(  # pylint: disable=R0913, R0914
    database: str,
    zeta: Optional[str] = None,
    filter_poles: bool = False,
    round_digits: int = 2,
    filter_by_nstates: bool = False,
    filter_degeneracy: bool = False,
    drop_comment: bool = True,
) -> pd.DataFrame:
    """Reads database and returns data frame with table plus additional columns.

    The database must contain an ``Energy`` table.
    Drops all entries which are at or close to divergencies of the zeta function.

    Argumennts:
        database: str
            The sqlite database file.

        zeta: Optional[str] = None
            Zeta function to use to convert spectrum to phase shifts.
            Valid options are 'spherical', 'dispersion' or 'cartesian'.
            Defaults to not converting (if ``None``).

        filter_poles: bool = False
            Automatically filter out states which are close to zeta poles.

        round_digits: int = 2
            How close an x-value has to be to an integer in order to be dropped.

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

    if filter_degeneracy:
        degenerate_nums = [n2 for n2, vec in get_degeneracy(20).items() if len(vec) > 1]
        df = df[~df.x.round(round_digits).isin(degenerate_nums)]

    if zeta is not None:
        df["y"] = get_ere(df, zeta=zeta)

    df = df.dropna()

    return df


def get_ere(df: pd.DataFrame, zeta: str = "spherical") -> pd.Series:
    """Converts data frame of box levels to effective range expansion.
    """
    if zeta == "spherical":
        s = spherical_zeta(df["x"].values) / df["L"]
    elif zeta == "dispersion":
        g = df.groupby(["L", "epsilon", "nstep"], as_index=False)
        s = (
            g.apply(
                lambda frame: pd.Series(
                    DispersionZeta3D(
                        frame["L"].unique()[0],
                        frame["epsilon"].unique()[0],
                        (
                            frame["nstep"].unique()[0]
                            if frame["nstep"].unique()[0] > 0
                            else None
                        ),
                    )(frame["x"].values)
                    / frame["L"],
                    index=frame.index,
                )
            )
            .reset_index(0)
            .drop(columns="level_0")
            .rename(columns={0: "y"})
        )
    elif zeta == "cartesian":
        g = df.groupby(["n1d"], as_index=False)
        s = (
            g.apply(
                lambda frame: pd.Series(
                    Zeta3D(frame["n1d"].mean())(frame["x"].values) / frame["L"],
                    index=frame.index,
                )
            )
            .reset_index(0)
            .drop(columns="level_0")
            .rename(columns={0: "y"})
        )
    else:
        raise KeyError("Recieved unkwon parater for zeta function (`{zeta}`)")

    return s / np.pi


def _poly(
    epsilon: np.ndarray, p: Dict[str, "GVar"], even: bool = True
):  # pylint: disable=E1101
    """Even polynomial used for fit.

    `y = sum(x_n epsilon**(2n), n=0, n_max)` if even else also sum over `epsilon**n`.

    **Arguments**
        epsilon: np.ndarray
            The lattice spacing to extrapolate.

        p: Dict[str, gv.GVar]
            The prior. Must contain key "x"

        even: bool = True
            Compute only even polynomial.
    """
    nexp = len(p["x"])
    exps = np.arange(0, nexp * 2 if even else nexp, 2 if even else 1)
    out = 0
    for exp, x in zip(exps, p["x"]):
        out += epsilon ** exp * x

    return out


def _group_wise_fit(  # pylint: disable=R0914
    row: pd.Series,
    n_poly_max: int = 4,
    delta_x=1.25e-13,
    include_statistics: bool = True,
    odd_poly: bool = False,
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

        odd_poly: bool = False
            Allow fits of odd polynomials.
    """
    epsilon, x = row["epsilon"].values, row["x"].values
    error = [delta_x * np.random.uniform(0.8, 1.2, size=1)[0] for _ in x]
    data = (epsilon, gv.gvar(x, error))

    poly = lambda ee, pp: _poly(ee, pp, even=not odd_poly)

    if len(x) > 2:

        fits = []
        log_gbf = []
        for nexp in range(1, n_poly_max + 1):
            x_init = np.zeros(nexp)
            x_init[0] = x[0]
            x_error = [10 * np.random.uniform(0.5, 2, size=1)[0] for _ in x_init]
            prior = {"x": gv.gvar(x_init, x_error)}

            fit = lsqfit.nonlinear_fit(data, fcn=poly, prior=prior)
            fits.append(fit)
            log_gbf.append(fit.logGBF)

        if include_statistics:
            data = []
            for fit in fits:
                tmp = {
                    "x": fit.p["x"][0].mean,
                    "sdev": fit.p["x"][0].sdev,
                    "chi2/dof": fit.chi2 / fit.dof,
                    "logGBF": fit.logGBF,
                    "n_poly_max": len(fit.p["x"]) - 1
                    if odd_poly
                    else 2 * (len(fit.p["x"]) - 1),
                    "even": not odd_poly,
                }
                for n, x in enumerate(fit.p["x"]):
                    tmp[f"x{n if odd_poly else 2*(n)}"] = x
                data.append(tmp)
            out = pd.DataFrame(data)
        else:
            idx = np.argmax(log_gbf)
            fit = fits[idx]
            out = pd.Series({"x": fit.p["x"][0].mean})
    else:
        out = None

    return out


def get_continuum_extrapolation(  # pylint: disable=C0103
    df: pd.DataFrame,
    n_poly_max: int = 4,
    delta_x: float = 1.25e-13,
    include_statistics: bool = True,
    odd_poly: bool = False,
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

        odd_poly: bool = False
            Allow fits of odd polynomials.
    """
    if lsqfit is None or gv is None:
        raise ImportError(
            "Cannort load `lsqfit` and `gvar`." " Thus fitting is not possible."
        )
    group = df.groupby(["L", "nstep", "nlevel"])[["epsilon", "x"]]
    fit_df = group.apply(
        _group_wise_fit,
        n_poly_max=n_poly_max,
        delta_x=delta_x,
        include_statistics=include_statistics,
        odd_poly=odd_poly,
    ).reset_index()
    fit_df["epsilon"] = 0

    if "level_3" in fit_df.columns:
        fit_df = fit_df.drop(columns=["level_3"])

    return fit_df
