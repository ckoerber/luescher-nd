"""Plotting utility functions for luescher_nd project
"""
from typing import List
from typing import Tuple

from itertools import product

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import seaborn as sns

from luescher_nd.zeta.zeta3d import DispersionZeta3d
from luescher_nd.zeta.extern import pyzeta

## Parameters & Files
HBARC = 197.326  # MeV / fm
M_NUCLEON = (938.27 + 939.57) / 2 / HBARC  # in fm^{-1}

MU = M_NUCLEON / 2


def plot_ere_grid(  # pylint: disable=C0103
    df: pd.DataFrame, x="x", y="y", **kwargs
) -> sns.FacetGrid:
    """Plots effective range expansion grid for input DataFrame.
    """
    default_kwargs = {
        "col": "$n_s$",
        "hue": "n1d_max",
        "hue_kws": {"marker": ["s", "o", "v", "d"]},
        "palette": "cubehelix",
        "margin_titles": True,
        "ylim": (-2, 2),
        "xlim": (-1, 10),
    }
    default_kwargs["col_order"] = sorted(df[default_kwargs["col"]].unique())

    default_kwargs.update(**kwargs)

    g = sns.FacetGrid(df, **default_kwargs)
    g.map(plt.plot, x, y, ms=2, lw=1, ls="--")

    for ax in np.array(g.axes).flatten():
        ax.axhline(0, c="grey", lw=1, zorder=-2)

    g.set_xlabels("$x$")
    g.set_ylabels(r"$p \cot (\delta_0(p))$")
    g.add_legend(title=r"$\frac{L}{\epsilon}$", fontsize=6)

    return g


def monotonous_bounds(vals: np.ndarray) -> List[Tuple[int, int]]:
    """Returns indices pairs of array such that each split is monotously increasing.
    """
    bounds = []
    prev_val = vals[0]
    prev_bound = 0
    for ind, val in enumerate(vals[1:]):
        if val < prev_val:
            bounds += [(prev_bound, ind + 1)]
            prev_bound = ind + 1
        prev_val = val
    bounds += [(prev_bound, len(vals))]

    return bounds


def raw_data_to_plot_frame(  # pylint: disable=C0103
    df: pd.DataFrame, dispersion_zeta: bool = True, ycut: float = 2.0
) -> pd.DataFrame:
    """Converts a raw data data frame to a plotting data frame.

    Adds columns and computes results of zeta function.

    **Arguments**
        df: pd.DataFrame
            Raw data input frame.

        dispersion_zeta: bool = True
            Use dispersion zeta function or standard zeta function.

        ycut: float = 2.0
            Drop data which has larger |y| than ycut.
    """
    zeta = {}

    df["L"] = df["n1d_max"] * df["epsilon"]
    df["x"] = df["energy"] * MU * df["L"] ** 2 / 2 / np.pi ** 2
    df.loc[df.nstep.isna(), "nstep"] = -1

    for epsilon, L, nstep in product(
        df.epsilon.unique(), df.L.unique(), df.nstep.unique()
    ):
        zeta[(epsilon, L, nstep)] = (
            DispersionZeta3d(L, epsilon, nstep=(None if nstep < 0 else nstep))
            if dispersion_zeta
            else pyzeta.zeta  # pylint: disable=I1101
        )

    df["y"] = df.apply(
        lambda row: zeta[(row["epsilon"], row["L"], row["nstep"])](row["x"])[0]
        / np.pi
        / row["L"],
        axis=1,
    )
    df.loc[df["y"].abs() > ycut, "y"] = np.nan
    df["$n_s$"] = df.apply(
        lambda row: r"$\infty$" if row["nstep"] < 0 else f"${row['nstep']}$", axis=1
    )

    df = (
        df.drop_duplicates(["epsilon", "nstep", "y"])
        .dropna()
        .reset_index(drop=True)
        .sort_values(["n1d_max", "nstep", "x", "nlevel"], ascending=True)
    )

    return df
